//
// Created by baojian on 6/15/18.
//

#ifndef PROJ_C_PAIRING_HEAP_H
#define PROJ_C_PAIRING_HEAP_H

#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct PHNode { // Pairing Heap Node
    double value;
    double child_offset;
    int payload;//key ?? TODO
    struct PHNode *child; // its first child
    struct PHNode *left_up; // to its parent in this binary tree
    struct PHNode *sibling; // point to its next older sibling
} PHNode;

typedef struct PairingHeap {
    size_t size;
    size_t hp_size;
    PHNode *root;// the minimum node
    int capacity;
    PHNode **heap;
} PairingHeap;


// use linking to combine two heap-ordered trees
static PHNode *_linking(PHNode *node1, PHNode *node2) {
    if (node1 == NULL) {
        return node2;
    }
    if (node2 == NULL) {
        return node1;
    }
    PHNode *small_node = node2;
    PHNode *large_node = node1;
    if (node1->value < node2->value) {
        small_node = node1;
        large_node = node2;
    }
    large_node->sibling = small_node->child;
    if (large_node->sibling != NULL) {
        large_node->sibling->left_up = large_node;
    }
    large_node->left_up = small_node;
    small_node->child = large_node;
    large_node->value -= small_node->child_offset;
    large_node->child_offset -= small_node->child_offset;
    return small_node;
}

/** create a new, empty heap named h */
PairingHeap *ph_make_heap(int capacity) {
    PairingHeap *ph = malloc(sizeof(PairingHeap));
    ph->size = 0;
    ph->hp_size = 0;
    ph->capacity = capacity;
    ph->root = NULL;
    ph->heap = (PHNode **) malloc(sizeof(PHNode *) * capacity);
    for (int i = 0; i < capacity; i++) {
        ph->heap[i] = (PHNode *) malloc(sizeof(PHNode));
        ph->heap[i]->sibling = NULL;
        ph->heap[i]->child = NULL;
        ph->heap[i]->left_up = NULL;
    }
    return ph;
}

/**to release all heap memory. */
void ph_free(PairingHeap *ph) {
    ph->hp_size = 0;
    if (ph->root != NULL) {
        ph->heap[ph->hp_size++] = ph->root;
    }
    size_t cur_ind = 0;
    while (cur_ind < ph->hp_size) {
        PHNode *cur_node = ph->heap[cur_ind];
        if (cur_node->child != NULL) {
            ph->heap[ph->hp_size++] = cur_node->child;
        }
        if (cur_node->sibling != NULL) {
            ph->heap[ph->hp_size++] = cur_node->sibling;
        }
        cur_ind += 1;
    }
    for (size_t ii = 0; ii < ph->hp_size; ++ii) {
        free(ph->heap[ii]);
    }
    free(ph->heap);
    ph->root = NULL;
    free(ph);
}

/** check if it is empty. */
extern inline bool hp_is_empty(PairingHeap *ph) {
    return ph->root == NULL;
}

/** return an item of minimum key from heap h, without changing h*/
bool ph_get_min(PairingHeap *ph, double *value, int *payload) {
    *value = ph->root->value;
    *payload = ph->root->payload;
    return true;
}

/** insert item data_x, with predefined key, into heap h,
 * not previously containing data_x*/
PHNode *ph_insert(PairingHeap *ph, double value, int payload) {
    PHNode *new_node = ph->heap[ph->hp_size++];
    new_node->value = value;
    new_node->payload = payload;
    new_node->child_offset = 0;
    ph->root = _linking(ph->root, new_node);
    ph->size++;
    return new_node;
}

void ph_add_to_heap(PairingHeap *ph, double value) {
    ph->root->value += value;
    ph->root->child_offset += value;
}

/** delete an item of minimum key from heap h and return it.
 * If h is originally empty, return a special null item.*/
bool ph_delete_min(PairingHeap *ph, double *value, int *payload) {
    printf("test1\n");
    if (ph->root == NULL) {
        return false;
    }
    printf("test2\n");
    PHNode *result = ph->root;
    ph->hp_size = 0;
    PHNode *cur_child = ph->root->child;
    PHNode *next_child = NULL;
    printf("test3\n");
    int index = 0;
    while (cur_child != NULL) {
        printf("test index: %d\n", index);
        ph->heap[ph->hp_size++] = cur_child;
        next_child = cur_child->sibling;
        cur_child->left_up = NULL;
        cur_child->sibling = NULL;
        cur_child->value += result->child_offset;
        cur_child->child_offset += result->child_offset;
        cur_child = next_child;
        index++;
    }
    size_t merged_children = 0;
    index = 0;
    while (merged_children + 2 <= ph->hp_size) {
        printf("test index__: %d\n", index++);
        ph->heap[merged_children / 2] = _linking(
                ph->heap[merged_children], ph->heap[merged_children + 1]);
        merged_children += 2;
    }
    printf("test3\n");
    printf("merged children: %ld, heap_size: %ld\n", merged_children,
           ph->hp_size);
    if (merged_children != ph->hp_size) {
        printf("test4\n");
        ph->heap[merged_children / 2] = ph->heap[merged_children];
        ph->hp_size = merged_children / 2 + 1;
    } else {
        ph->hp_size = merged_children / 2;
        printf("test4\n");
    }
    if (ph->hp_size > 0) {
        printf("test5\n");
        ph->root = ph->heap[ph->hp_size - 1];
        for (size_t ii = ph->hp_size - 2; ii >= 0; --ii) {
            ph->root = _linking(ph->root, ph->heap[ii]);
        }
    } else {
        ph->root = NULL;
    }
    printf("final");
    *value = result->value;
    *payload = result->payload;
    return true;
}

static PairingHeap *ph_meld(PairingHeap *hp1, PairingHeap *hp2) {
    PairingHeap *re_ph = (PairingHeap *) malloc(sizeof(PairingHeap));
    re_ph->root = _linking(hp1->root, hp2->root);
    hp1->root = NULL;
    hp2->root = NULL;
    return re_ph;
}

/** decrease the key of item x in heap h by subtracting the non-negative
 * real number delta*/
PHNode *ph_decrease_key(PairingHeap *ph, PHNode *node, double from_value,
                        double to_value) {
    double additional_offset = from_value - node->value;
    node->child_offset += additional_offset;
    node->value = to_value;
    if (node->left_up != NULL) {
        if (node->left_up->child == node) {
            node->left_up->child = node->sibling;
        } else {
            node->left_up->sibling = node->sibling;
        }
        if (node->sibling != NULL) {
            node->sibling->left_up = node->left_up;
        }
        node->left_up = NULL;
        node->sibling = NULL;
        ph->root = _linking(ph->root, node);
    }
}

#endif //PROJ_C_PAIRING_HEAP_H
