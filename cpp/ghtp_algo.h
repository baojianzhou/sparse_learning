//
// Created by baojian on 7/15/18.
//

#ifndef SPARSE_PROJ_GHTP_ALGO_H
#define SPARSE_PROJ_GHTP_ALGO_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cblas.h>


struct data_ {
    double val;
    int ori_index;
};


extern inline double *expit(const double *x, double *out, int len_x) {
    for (int i = 0; i < len_x; i++) {
        if (x[i] > 0) {
            out[i] = 1. / (1. + exp(-x[i]));
        } else {
            out[i] = 1. - 1. / (1. + exp(x[i]));
        }
    }
}


void loss_logistic_sigmoid(const double *x, double *out, int x_len) {
    for (int i = 0; i < x_len; i++) {
        if (x[i] > 0.0) {
            out[i] = -log(1.0 + exp(-x[i]));
        } else {
            out[i] = x[i] - log(1.0 + exp(x[i]));
        }
    }
}


/**
 * Computes the logistic loss and gradient.
 * Let n_samples=n, n_features=p
 * Parameters
 * ----------
 * @param w: (p+1,)     coefficient vector + intercept.
 * @param x: (n, p)     training data.
 * @param y: (n,)       array of labels.
 * @param eta: float    l2-regularization parameter.
 * @param n_samples:    n: number of samples
 * @param n_features:   p: number of features
 * @return (p+2,) loss, grad, intercept
 */
double *loss_logistic_loss_grad(const double *w,
                                const double *x,
                                const double *y,
                                double *loss_grad,
                                double eta,
                                int n_samples,
                                int n_features) {
    int i, n = n_samples, p = n_features;
    double intercept = w[p], sum_z0 = 0.0;
    auto *yz = (double *) malloc(n * sizeof(double));
    auto *z0 = (double *) malloc(n * sizeof(double));
    auto *logistic = (double *) malloc(n * sizeof(double));
    auto *ones = (double *) malloc(n * sizeof(double));
    for (i = 0; i < n; i++) { /** calculate yz */
        yz[i] = intercept;
        ones[i] = 1.;
    }
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, p, 1., x, p, w, 1, 1., yz, 1);
    for (i = 0; i < n; i++) {
        yz[i] *= y[i];
    }
    /** calculate z0 (final intercept)*/
    expit(yz, z0, n);
    for (i = 0; i < n; i++) {
        z0[i] = (z0[i] - 1.) * y[i];
        sum_z0 += z0[i];
    }
    /** calculate loss */
    loss_logistic_sigmoid(yz, logistic, n);
    loss_grad[0] = 0.5 * eta * cblas_ddot(p, w, 1, w, 1);   // regularization
    loss_grad[0] -= cblas_ddot(n, ones, 1, logistic, 1);    // data fitting
    /** calculate gradient*/
    memcpy(loss_grad + 1, w, sizeof(double) * p);
    cblas_dgemv(CblasRowMajor, CblasTrans, n, p, 1., x, p, z0, 1,
                eta, loss_grad + 1, 1);
    loss_grad[p] = sum_z0; // intercept part
    free(ones);
    free(logistic);
    free(z0);
    free(yz);
    return loss_grad;
}

extern inline int data_compare(const void *a, const void *b) {
    if (((struct data_ *) a)->val < ((struct data_ *) b)->val) {
        return 1;
    } else if (((struct data_ *) a)->val == ((struct data_ *) b)->val) {
        return 0;
    } else {
        return -1;
    }
}

bool argsort(double *w, int s, int w_len, int *set_s) {
    auto *w_tmp = (struct data_ *) malloc(sizeof(data_) * w_len);
    for (int i = 0; i < w_len; i++) {
        w_tmp[i].val = abs(w[i]);
        w_tmp[i].ori_index = i;
    }
    qsort(w_tmp, static_cast<size_t>(w_len),
          sizeof(struct data_), &data_compare);
    for (size_t i = 0; i < s; i++) {
        set_s[i] = w_tmp[i].ori_index;
    }
    set_s[s] = w_len;
    free(w_tmp);
    return true;
}

double *min_f(const int *set_s, const double *x_tr,
              const double *y_tr, const double *w0,
              int max_iter, double eta, double *wt,
              int n, int p, int s) {
    int i;
    auto *loss_grad = (double *) malloc((p + 2) * sizeof(double));
    auto *wt_tmp = (double *) malloc((p + 1) * sizeof(double));
    auto *tmp_loss_grad = (double *) malloc((p + 2) * sizeof(double));
    // make sure start point is a feasible point.
    cblas_dcopy(p + 1, w0, 1, wt, 1); // starts from initial point
    // a Frank-Wolfe style minimization with backtracking line search
    double beta, lr, grad_sq;
    for (i = 0; i < max_iter; i++) {
        loss_logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, n, p);
        beta = 0.8, lr = 1.0;
        grad_sq = 0.5 * cblas_ddot(p + 1, loss_grad + 1, 1, loss_grad + 1, 1);
        while (true) { // using backtracking line search
            cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
            cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
            loss_logistic_loss_grad(wt_tmp, x_tr, y_tr,
                                    tmp_loss_grad, eta, n, p);
            if (tmp_loss_grad[0] > (loss_grad[0] - lr * grad_sq)) {
                lr = beta * lr;
            } else {
                break;
            }
        }
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt, 1);
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int k = 0; k < (s + 1); k++) {
            wt_tmp[set_s[k]] = wt[set_s[k]];
        }
        cblas_dcopy(p, wt_tmp, 1, wt, 1);
    }
    free(tmp_loss_grad);
    free(loss_grad);
}


#endif //SPARSE_PROJ_GHTP_ALGO_H
