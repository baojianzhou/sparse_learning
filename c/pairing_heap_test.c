//
// Created by baojian on 6/15/18.
//
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "pairing_heap.h"

void test() {
    int i;
    typedef struct Test {
        int data;
        int type;
    } TestType;
    TestType **p = malloc(10 * sizeof(TestType *));
    for (i = 0; i < 10; i++) {
        p[i] = (TestType *) malloc(sizeof(TestType));
        p[i]->data = i;
        p[i]->type = i + 10;
    }
    TestType ***ppp = &p;
    for (i = 0; i < 10; i++) {
        printf("%d %d\n", p[i]->data, p[i]->type);
        printf("%d %d\n", (*ppp)[i]->data, (*ppp)[i]->type);
    }
    time_t t;
    printf("test pairing heap_item_handle\n");
}

int main(int argc, char *argv[]) {
    int i;
    time_t t;
    printf("test pairing heap_item_handle\n");
    srand((unsigned) time(&t));
    PairingHeap *ph = ph_make_heap(10);
    ph_print_heap(ph);
    for (i = 0; i < 10; i++) {
        double rand_val = (rand() % 100) * 897;
        int pay_load = rand() % 100;
        printf("insert val: %lf, payload: %d\n", rand_val, pay_load);
        ph_insert(ph, rand_val, pay_load);
    }
    ph_print_heap(ph);
    double *min_val = (double *) malloc(sizeof(double));
    int *min_payload = (int *) malloc(sizeof(int));
    ph_get_min(ph, min_val, min_payload);
    printf("minimal node val: %lf payload: %d\n", *min_val, *min_payload);
    for (i = 0; i < 9; i++) {
        ph_delete_min(ph, min_val, min_payload);
        printf("delete val: %lf, payload: %d\n", *min_val, *min_payload);
    }
    printf("how many left: %ld\n", ph->size);
    ph_free(ph);
    printf("is empty: %d\n", hp_is_empty(ph));
}