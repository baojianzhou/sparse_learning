//
// Created by baojian on 7/15/18.
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cblas.h>

int main() {
    int p = 5;
    auto *w = (double *) malloc(sizeof(double) * 5);
    w[0] = 3.;
    w[1] = 4.;
    w[2] = 5.;
    w[3] = 6.;
    w[4] = 7.;
    auto *loss_grad = (double *) malloc(6 * sizeof(double));
    loss_grad[0] = 2.;
    for (int i = 0; i < 6; i++) {
        printf("%lf ", loss_grad[i]);
    }
    printf("\n");
    memcpy(loss_grad + 1, w, sizeof(double) * p);
    for (int i = 0; i < 6; i++) {
        printf("%lf ", loss_grad[i]);
    }
    printf("\n");
}
