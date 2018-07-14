//
// Created by baojian on 6/15/18.
//
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "pairing_heap.h"

int main(int argc, char *argv[]) {
    int i;
    time_t t;
    printf("test pairing heap\n");
    srand((unsigned) time(&t));
    PairingHeap *ph = ph_make_heap(10);
    if (hp_is_empty(ph)) {
        printf("hp is empty.\n");
    } else {
        printf("hp is not empty.\n");
    }
    printf("capacity of the heap: %d\n", ph->capacity);
    for (i = 0; i < 10; i++) {
        double rand_val = (rand() % 100) * 897;
        int pay_load = rand() % 100;
        printf("insert val: %lf, payload: %d\n", rand_val, pay_load);
        ph_insert(ph, rand_val, pay_load);
    }
    printf("is empty: %d\n", hp_is_empty(ph));
    printf("heap_size of this heap: %ld\n", ph->hp_size);
    double *min_val = (double *) malloc(sizeof(double));
    int *min_payload = (int *) malloc(sizeof(int));
    ph_get_min(ph, min_val, min_payload);
    printf("minimal node val: %lf payload: %d\n", *min_val, *min_payload);
    ph_add_to_heap(ph, 10000.);
    printf("add 10. to heap.\n");
    ph_get_min(ph, min_val, min_payload);
    printf("minimal node val: %lf payload: %d\n", *min_val, *min_payload);
    for (i = 0; i < 9; i++) {
        ph_delete_min(ph, min_val, min_payload);
        printf("delete val: %lf, payload: %d\n", *min_val, *min_payload);
    }
    printf("how many left: %ld\n", ph->hp_size);
    ph_free(ph);
    printf("is empty: %d\n", hp_is_empty(ph));
}