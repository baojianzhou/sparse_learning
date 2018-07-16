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


extern inline void expit(const double *x, double *out, int len_x) {
    for (int i = 0; i < len_x; i++) {
        if (x[i] > 0) {
            out[i] = 1. / (1. + exp(-x[i]));
        } else {
            out[i] = 1. - 1. / (1. + exp(x[i]));
        }
    }
}


extern inline void log_logistic(const double *x, double *out, int x_len) {
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
 * @param w: (p+1,)         coefficient vector + intercept (last element w[p]).
 * @param x: (n, p)         training data.
 * @param y: (n,)           array of labels.
 * @param loss_grad: (p+2,) to save the result of loss, grad, and intercept.
 * @param eta: float        l2-regularization parameter.
 * @param n_samples:        n: number of samples
 * @param n_features:       p: number of features
 * @return (p+2,)           loss, grad, intercept
 */
void logistic_loss_grad(const double *w, const double *x,
                        const double *y, double *loss_grad,
                        double eta, int n_samples, int n_features) {
    int i, n = n_samples, p = n_features;
    double intercept = w[p], sum_z0 = 0.0;
    loss_grad[0] = 0.0;
    auto *yz = (double *) malloc(n * sizeof(double));
    auto *z0 = (double *) malloc(n * sizeof(double));
    auto *logistic = (double *) malloc(n * sizeof(double));
    for (i = 0; i < n; i++) { /** calculate yz */
        yz[i] = intercept;
    }
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, p, 1., x, p, w, 1, 1., yz, 1);
    for (i = 0; i < n; i++) {
        yz[i] *= y[i];
    }
    /** calculate z0 and final intercept*/
    expit(yz, z0, n);
    /** calculate logistic logistic[i] = 1/(1+exp(-y[i]*(xi^T*w+c)))*/
    log_logistic(yz, logistic, n);
    /** calculate loss of data fitting part*/
    for (i = 0; i < n; i++) {
        z0[i] = (z0[i] - 1.) * y[i];
        sum_z0 += z0[i];
        loss_grad[0] -= logistic[i];
    }
    /**calculate loss of regularization part (it does not have intercept)*/
    loss_grad[0] += 0.5 * eta * cblas_ddot(p, w, 1, w, 1);
    /** calculate gradient of coefficients*/
    memcpy(loss_grad + 1, w, sizeof(double) * p);
    /** x^T*z0 + eta*w, where z0[i]=(logistic[i] - 1.)*yi*/
    cblas_dgemv(CblasRowMajor, CblasTrans, n, p, 1., x, p, z0, 1,
                eta, loss_grad + 1, 1);
    /** calculate gradient of intercept part*/
    loss_grad[p + 1] = sum_z0; // intercept part
    free(logistic);
    free(z0);
    free(yz);
}

extern inline int comp(const void *a, const void *b) {
    if (((struct data_ *) a)->val < ((struct data_ *) b)->val) {
        return 1;
    } else if (((struct data_ *) a)->val == ((struct data_ *) b)->val) {
        return 0;
    } else {
        return -1;
    }
}

void argsort(double *w, int s, int p, int *set_s) {
    auto *w_tmp = (struct data_ *) malloc(sizeof(data_) * p);
    for (int i = 0; i < p; i++) {
        w_tmp[i].val = abs(w[i]);
        w_tmp[i].ori_index = i;
    }
    qsort(w_tmp, static_cast<size_t>(p), sizeof(struct data_), &comp);
    for (size_t i = 0; i < s; i++) {
        set_s[i] = w_tmp[i].ori_index;
    }
    free(w_tmp);
}

void min_f(const int *set_s, const double *x_tr,
           const double *y_tr, int max_iter, double eta, double *wt,
           int n, int p, int set_s_len) {
    int i;
    auto *loss_grad = (double *) malloc((p + 2) * sizeof(double));
    auto *tmp_loss_grad = (double *) malloc((p + 2) * sizeof(double));
    auto *wt_tmp = (double *) malloc((p + 1) * sizeof(double));
    /**
     * make sure start point is a feasible point. here we do a trick:
     * we treat wt as an initial point itself. and of course wt is always a
     * feasible point. A Frank-Wolfe style minimization with
     * backtracking line search. Other algorithms can be considered:
     * Newton's method, trust region, etc.
     */
    double beta, lr, grad_sq;
    for (i = 0; i < max_iter; i++) {
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, n, p);
        beta = 0.8, lr = 1.0;
        grad_sq = 0.5 * cblas_ddot(p + 1, loss_grad + 1, 1, loss_grad + 1, 1);
        while (true) { // using backtracking line search
            cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
            cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
            logistic_loss_grad(wt_tmp, x_tr, y_tr, tmp_loss_grad, eta, n, p);
            if (tmp_loss_grad[0] > (loss_grad[0] - lr * grad_sq)) {
                lr *= beta;
            } else {
                break;
            }
        }
        // projection step
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int k = 0; k < set_s_len; k++) {
            wt_tmp[set_s[k]] = wt[set_s[k]];
        }
        wt_tmp[p] = wt[p];
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
    }
    free(wt_tmp);
    free(tmp_loss_grad);
    free(loss_grad);
}


void min_f_constrained(const int *set_s, const double *x_tr,
                       const double *y_tr, int max_iter, double eta,
                       double *wt, int n, int p, int set_s_len) {
    int i, j, p_ = set_s_len;
    auto *x_tr_ = (double *) malloc(n * p_ * sizeof(double));
    auto *loss_grad = (double *) malloc((p_ + 2) * sizeof(double));
    auto *tmp_loss_grad = (double *) malloc((p_ + 2) * sizeof(double));
    auto *wt_tmp = (double *) malloc((p_ + 1) * sizeof(double));
    auto *c_wt = (double *) malloc((p_ + 1) * sizeof(double));
    /**
     * make sure start point is a feasible point. here we do a trick:
     * we treat wt as an initial point itself. and of course wt is always a
     * feasible point. A Frank-Wolfe style minimization with
     * backtracking line search. Other algorithms can be considered:
     * Newton's method, trust region, etc.
     */
    //get sub-matrix with set_s constraint
    for (i = 0; i < n; i++) {
        for (j = 0; j < p_; j++) {
            x_tr_[i * p_ + j] = x_tr[i * p + set_s[j]];
        }
    }
    for (i = 0; i < p_; i++) {
        c_wt[i] = wt[set_s[i]];
    }
    c_wt[p_] = wt[p];
    double beta, lr, grad_sq;
    for (i = 0; i < max_iter; i++) {
        logistic_loss_grad(c_wt, x_tr_, y_tr, loss_grad, eta, n, p_);
        beta = 0.8, lr = 1.0;
        grad_sq = 0.5 * cblas_ddot(p_ + 1, loss_grad + 1, 1, loss_grad + 1, 1);
        while (true) { // using backtracking line search
            cblas_dcopy(p_ + 1, c_wt, 1, wt_tmp, 1);
            cblas_daxpy(p_ + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
            logistic_loss_grad(wt_tmp, x_tr, y_tr, tmp_loss_grad, eta, n, p_);
            if (tmp_loss_grad[0] > (loss_grad[0] - lr * grad_sq)) {
                lr *= beta;
            } else {
                break;
            }
        }
        cblas_dcopy(p_ + 1, wt_tmp, 1, c_wt, 1);
    }
    // projection step
    cblas_dscal(p + 1, 0.0, wt, 1);
    for (i = 0; i < p_; i++) {
        wt[set_s[i]] = c_wt[i];
    }
    wt[p] = c_wt[p_];
    free(c_wt);
    free(wt_tmp);
    free(tmp_loss_grad);
    free(loss_grad);
    free(x_tr_);
}


#endif //SPARSE_PROJ_GHTP_ALGO_H
