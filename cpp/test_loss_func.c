//
// Created by baojian on 6/29/18.
//

#include "loss_func.h"
#include <stdio.h>

double test_func() {
    double w[3] = {1, 2, 3};
    for (int i = 0; i < 3; i++) {
        printf("w[%d]: %lf", i, w[i]);
    }

}

int main() {
    test_func();
}