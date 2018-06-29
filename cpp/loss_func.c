//
// Created by baojian on 6/29/18.
//
#include <math.h>
#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <string.h>
#include "loss_func.h"

double loss_logistic_primal_loss(const double w_xi,
                                 const double yi, const double weight) {
    // Logistic loss:
    //   log(1 + e^(-ywx))
    //   log(e^0 + e^(-ywx))
    //   a + log(e^(0-a) + e^(-ywx - a)),  where a is max(0, -ywx)
    // https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
    const double y_wx = yi * w_xi;
    if (y_wx > 0) {
        // 0 + log(e^(0) + e^(-ywx - 0))
        // log(1 + e^(-ywx))
        return log(1 + exp(-y_wx)) * weight;
    }
    // -ywx + log(e^(ywx) + e^(-ywx + ywx))
    // log(e^(ywx) + e^(0)) - ywx
    // log(1 + e^(ywx)) - ywx
    return (log(1 + exp(y_wx)) - y_wx) * weight;
}

double loss_logistic_primal_derivative(const double w_xi, const double yi,
                                       const double weight) {
    double inverse_exp_term = 0;
    if (yi * w_xi > 0) {
        inverse_exp_term = exp(-yi * w_xi) / (1 + exp(-yi * w_xi));
    } else {
        inverse_exp_term = 1 / (1 + exp(yi * w_xi));
    }
    return inverse_exp_term * yi * weight;
}


double *loss_logistic_intercept_dot(const double *w,
                                    const double *x,
                                    const double *y,
                                    const int w_len,
                                    const int x_len,
                                    const int y_len) {
    double intercept = 0.0;
    int p = x_len / y_len;
    if ((p + 1) == w_len) {
        intercept = w[w_len - 1];
    }
    double *x_dot_w = (double *) malloc(y_len * sizeof(double));
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                y_len, p, 1, x, y_len, w, 1, 0., x_dot_w, 1);
}

double *loss_logistic_loss_and_grad(const double *w,
                                    const double *x,
                                    const double *y,
                                    double alpha,
                                    const double *weight,
                                    int w_len,
                                    int n_samples,
                                    int n_features) {
    double *grad = (double *) malloc(w_len * sizeof(double));
    double intercept = 0.0;
    if ((n_features + 1) == w_len) {
        intercept = w[w_len - 1];
    }
    double *x_dot_w = (double *) malloc(n_samples * sizeof(double));

    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                n_samples, n_features, 1, x, n_samples, w, 1, 0., x_dot_w, 1);
}

double *loss_logistic_loss(const double *w,
                           const double *x,
                           const double *y,
                           double alpha,
                           const double *weight,
                           int w_len,
                           int n_samples,
                           int n_features) {
    double loss = 0.0;
    double c = 0.0;
    if ((n_features + 1) == w_len) {
        c = w[w_len - 1];
    }
    double *x_dot_w = (double *) malloc(n_samples * sizeof(double));
    memset(x_dot_w, 1, (size_t) n_features);
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                n_samples, n_features, 1, x, n_samples, w, 1, 0., x_dot_w, 1);
    if (weight == NULL) {

    }
}
