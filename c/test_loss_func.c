//
// Created by baojian on 6/29/18.
//
#include <time.h>
#include <math.h>
#include <cblas.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "loss_func.h"

double sample_normal() {
    double u = ((double) random() / (RAND_MAX)) * 2 - 1;
    double v = ((double) random() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sample_normal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}

void test_logistic_sigmoid() {
    printf("method_1 logistic sigmoid ---\n");
    time_t t;
    srand((unsigned) time(&t));
    int n = 100, p = 200;
    double *x = (double *) malloc(n * p * sizeof(double));
    double *out = (double *) malloc(n * p * sizeof(double));
    for (int i = 0; i < n * p; i++) {
        x[i] = ((double) random() / (RAND_MAX));
    }
    clock_t begin = clock();
    loss_logistic_sigmoid(x, out, n * p);
    clock_t end = clock();
    double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    printf("time %lf seconds\n", time_spent);
    free(x);
    free(out);
    double xx[6] = {1., 0.5, 1., 2., 0.0, -1.};
    double outt[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    loss_logistic_sigmoid(xx, outt, 6);
    for (int i = 0; i < 6; i++) {
        printf("%.20lf\n", outt[i]);
    }
}


void test_logistic_loss() {
    printf("method_1 logistic loss ---\n");
    time_t t;
    srand((unsigned) time(&t));
    int i, n = 100, p = 100;
    double *w = (double *) malloc(p * sizeof(double));
    double *x = (double *) malloc(n * p * sizeof(double));
    double *y = (double *) malloc(n * sizeof(double));
    clock_t begin = clock();
    for (i = 0; i < n * p; i++) {
        x[i] = sample_normal();
    }
    for (i = 0; i < p; i++) {
        w[i] = sample_normal();
    }
    for (i = 0; i < n; i++) {
        if ((double) (random()) / (double) (RAND_MAX) < 0.5) {
            y[i] = 1;
        } else {
            y[i] = -1;
        }
    }
    clock_t end = clock();
    double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    printf("time of generate random is %lf seconds\n", time_spent);
    begin = clock();
    double loss = loss_logistic_loss(w, x, y, 0.5, NULL, p, n, p);
    end = clock();
    time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    printf("time of grad %lf seconds loss: %lf\n", time_spent, loss);
    free(y);
    free(x);
    free(w);
}

void test_logistic_loss_grad() {
    time_t t;
    srand((unsigned) time(&t));
    int i, n = 1000, p = 1000;
    double *w = (double *) malloc(p * sizeof(double));
    double *x = (double *) malloc(n * p * sizeof(double));
    double *y = (double *) malloc(n * sizeof(double));
    clock_t begin = clock();
    for (i = 0; i < n * p; i++) {
        x[i] = sample_normal();
    }
    for (i = 0; i < p; i++) {
        w[i] = sample_normal();
    }
    for (i = 0; i < n; i++) {
        if ((double) (random()) / (double) (RAND_MAX) < 0.5) {
            y[i] = 1;
        } else {
            y[i] = -1;
        }
    }
    clock_t end = clock();
    double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    printf("time %lf seconds\n", time_spent);
    begin = clock();
    double *loss_grad = loss_logistic_loss_grad(w, x, y, 0.5, NULL, p, n, p);
    end = clock();
    time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    printf("time %lf seconds\n", time_spent);
    for (i = 0; i < 10; i++) {
        printf("%lf\n", loss_grad[i]);
    }
    printf("\n");
    free(x);
    free(y);
    free(w);
}

void main() {
    test_logistic_sigmoid();
    test_logistic_loss();
    test_logistic_loss_grad();
}