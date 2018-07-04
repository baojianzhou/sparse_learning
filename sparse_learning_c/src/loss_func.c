//
// Created by baojian on 6/29/18.
//
#include <math.h>
#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <string.h>
#include "loss_func.h"


extern inline double *self_ddot(const double *x, double *out, int len_x) {
    for (int i = 0; i < len_x; i++) {
        out[i] *= x[i];
    }
}

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


double *loss_logistic_loss_grad(const double *w,
                                const double *x,
                                const double *y,
                                double alpha,
                                const double *weight,
                                int w_len,
                                int n_samples,
                                int n_features) {
    int i, n = n_samples, p = n_features;
    double c = 0.0, sum_z0 = 0.0;
    double *loss_grad = (double *) malloc((w_len + 1) * sizeof(double));
    double *unit_weight = (double *) malloc(n * sizeof(double));
    double *yz = (double *) malloc(n * sizeof(double));
    double *z0 = (double *) malloc(n * sizeof(double));
    double *logistic = (double *) malloc(n * sizeof(double));
    /** set intercept c */
    if ((p + 1) == w_len) {
        c = w[w_len - 1];
    }
    /** set weight */
    if (weight == NULL) {
        for (i = 0; i < n; i++) {
            unit_weight[i] = 1.;
        }
    } else {
        for (i = 0; i < n; i++) {
            unit_weight[i] = weight[i];
        }
    }
    /** calculate yz */
    for (i = 0; i < n; i++) {
        yz[i] = c;
    }
    cblas_dgemv(101, 111, n, p, 1., x, p, w, 1, 1., yz, 1);
    self_ddot(y, yz, n);
    /** calculate z0*/
    expit(yz, z0, n);
    for (i = 0; i < n; i++) {
        z0[i] = unit_weight[i] * (z0[i] - 1.) * y[i];
        sum_z0 += z0[i];
    }
    /** calculate loss */
    loss_logistic_sigmoid(yz, logistic, n);
    loss_grad[0] = 0.5 * alpha * cblas_ddot(p, w, 1, w, 1); // regularization
    loss_grad[0] -= cblas_ddot(n, unit_weight, 1, logistic, 1); // data fitting
    /** calculate gradient*/
    memcpy(loss_grad + 1, w, sizeof(double) * p);
    cblas_dgemv(101, 112, n, p, 1., x, p, z0, 1, alpha, loss_grad + 1, 1);
    if ((p + 1) == w_len) {
        loss_grad[w_len] = sum_z0; // intercept part
    }
    free(logistic);
    free(z0);
    free(yz);
    free(unit_weight);
    return loss_grad;
}

double loss_logistic_loss(const double *w,
                          const double *x,
                          const double *y,
                          double alpha,
                          const double *weight,
                          int w_len,
                          int n_samples,
                          int n_features) {
    int i, n = n_samples, p = n_features;
    double c = 0.0, loss;
    double *unit_weight = (double *) malloc(n * sizeof(double));
    double *yz = (double *) malloc(n * sizeof(double));
    double *logist = (double *) malloc(n * sizeof(double));
    /** set intercept c */
    if ((p + 1) == w_len) {
        c = w[w_len - 1];
    }
    /** set weight */
    if (weight == NULL) {
        for (i = 0; i < n; i++) {
            unit_weight[i] = 1.;
        }
    } else {
        for (i = 0; i < n; i++) {
            unit_weight[i] = weight[i];
        }
    }
    /** calculate yz */
    for (i = 0; i < n; i++) {
        yz[i] = c;
    }
    cblas_dgemv(101, 111, n, p, 1., x, p, w, 1, 1., yz, 1);
    self_ddot(y, yz, n);
    /** calculate loss */
    loss_logistic_sigmoid(yz, logist, n);
    loss = 0.5 * alpha * cblas_ddot(p, w, 1, w, 1); // regularization
    loss -= cblas_ddot(n, unit_weight, 1, logist, 1); // data fitting
    free(logist);
    free(yz);
    free(unit_weight);
    return loss;
}