//
// Created by baojian on 6/29/18.
// command: gcc -o test_loss_func test_loss_func.c loss_func.c loss_func.h -lm
// -I /usr/include/x86_64-linux-gnu/
// -L /usr/lib/x86_64-linux-gnu/openblas/
// -lopenblas -lpthread -lgfortran
//

#include <math.h>
#include <cblas.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "loss_func.h"

void test_intercept_dot() {
    int i = 0;
    double w[4] = {1.0, 2.0, 1.0, 1.0};
    double x[9] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0, -3.0, 4.0, -1.0};
    double y[3] = {1, -1, 1};
    double *result = loss_logistic_intercept_dot(w, x, y, 4, 9, 3);
    for (i = 0; i < 3; i++) {
        printf("%lf \n", result[i]);
    }
    free(result);
}

double sampleNormal() {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sampleNormal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}

void test_logistic_loss() {
    time_t t;
    srand((unsigned) time(&t));
    double *x = (double *) malloc(100000000 * sizeof(double));
    double *out = (double *) malloc(100000000 * sizeof(double));
    for (int i = 0; i < 10; i++) {
        x[i] = sampleNormal();
        printf("%lf\n", x[i]);
    }
    clock_t begin = clock();
    loss_logistic_sigmoid(x, out, 100000000);
    clock_t end = clock();
    double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    printf("time %lf seconds", time_spent);
    for (int i = 0; i < 10; i++) {
        printf("%lf\n", out[i]);
    }
    printf("\n");
    free(x);
    free(out);
    double xx[6] = {1., 0.5, 1., 2., 0.0, -1.};
    double outt[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    loss_logistic_sigmoid(xx, outt, 6);
    for (int i = 0; i < 6; i++) {
        printf("%.20lf\n", outt[i]);
    }
}

void test_logistic_loss_grad() {
    time_t t;
    srand((unsigned) time(&t));
    int i, n = 10000, p = 10000;

    double *w = (double *) malloc(p * sizeof(double));
    double *x = (double *) malloc(n * p * sizeof(double));
    double *y = (double *) malloc(n * sizeof(double));
    clock_t begin = clock();
    for (i = 0; i < n * p; i++) {
        x[i] = sampleNormal();
    }
    for (i = 0; i < p; i++) {
        w[i] = sampleNormal();
    }
    for (i = 0; i < n; i++) {
        if ((double) (rand()) / (double) (RAND_MAX) < 0.5) {
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
    for (int i = 0; i < 10; i++) {
        printf("%lf\n", loss_grad[i]);
    }
    printf("\n");
    free(x);
    free(y);
    free(w);
}

void main() {
    int n = 2, p = 5;
    double w[6] = {0., 0., 0., 0., -0.0};
    double x[10] = {1., 2., 3., 4., -2, 1., 2., 3., 4., -.3};
    double y[2] = {1., -1.};
    double *loss_grad = loss_logistic_loss_grad(w, x, y, 0.5, NULL, p + 1, n,
                                                p);
    for (int i = 0; i < 6; i++) {
        printf("%lf\n", loss_grad[i]);
    }
}