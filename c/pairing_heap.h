#ifndef PROJ_C_PAIRING_HEAP_H
#define PROJ_C_PAIRING_HEAP_H

#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct PHNode {     // Pairing Heap Node
    double value;
    double child_offset;
    int payload;            // key ?? TODO
    struct PHNode *child;   // its first child
    struct PHNode *left_up; // to its parent in this binary tree
    struct PHNode *sibling; // point to its next older sibling
} PHNode;

typedef struct PairingHeap {
    size_t size;            // current number of nodes inserted in this heap
    PHNode *root;           // the minimum node
    size_t capacity;        // maximal # of nodes can handle by this heap
    PHNode **heap_item_handle;
    PHNode *heap_buffer;
} PairingHeap;


// use linking to combine two heap_item_handle-ordered trees
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
PairingHeap *ph_make_heap(size_t capacity) {
    PairingHeap *ph = malloc(sizeof(PairingHeap));
    ph->size = 0;
    ph->capacity = capacity;
    ph->root = NULL;
    ph->heap_item_handle = malloc(capacity * sizeof(PHNode *));
    ph->heap_buffer = malloc(capacity * sizeof(PHNode));
    for (int i = 0; i < capacity; i++) {
        ph->heap_item_handle[i] = &ph->heap_buffer[i];
        ph->heap_buffer[i].value = -1.0;
        ph->heap_buffer[i].payload = 1;
        ph->heap_buffer[i].child_offset = -1.0;
        ph->heap_buffer[i].sibling = NULL;
        ph->heap_buffer[i].child = NULL;
        ph->heap_buffer[i].left_up = NULL;
    }
    return ph;
}

/**to release all heap memory. */
void ph_free(PairingHeap *ph) {
    free(ph->heap_buffer);
    free(ph->heap_item_handle);
    free(ph);
}

/** check if it is empty. */
extern inline bool hp_is_empty(PairingHeap *ph) {
    return ph->root == NULL;
}

/** return an item of minimum key from heap h, without changing h*/
bool ph_get_min(PairingHeap *ph, double *value, int *payload) {
    if (ph->root != NULL) {
        *value = ph->root->value;
        *payload = ph->root->payload;
        return true;
    }
    return true;
}

/** insert item data_x, with predefined key, into heap h,
 * not previously containing data_x*/
PHNode *ph_insert(PairingHeap *ph, double value, int payload) {
    PHNode *new_node = &ph->heap_buffer[ph->size++];
    new_node->sibling = NULL;
    new_node->child = NULL;
    new_node->left_up = NULL;
    new_node->value = value;
    new_node->payload = payload;
    new_node->child_offset = 0;
    ph->root = _linking(ph->root, new_node);
    return new_node;
}

void ph_add_to_heap(PairingHeap *ph, double value) {
    if (ph->root != NULL) {
        ph->root->value += value;
        ph->root->child_offset += value;
    }
}

/** delete an item of minimum key from heap h and return it.
 * If h is originally empty, return a special null item.*/
bool ph_delete_min(PairingHeap *ph, double *value, int *payload) {
    if (ph->root == NULL) {
        return false;
    }
    PHNode *result = ph->root;
    ph->size = 0;
    PHNode *cur_child = ph->root->child;
    PHNode *next_child = NULL;
    while (cur_child != NULL) {
        ph->heap_item_handle[ph->size++] = cur_child;
        printf("val1: %lf\n", cur_child->value);
        next_child = cur_child->sibling;
        cur_child->left_up = NULL;
        cur_child->sibling = NULL;
        cur_child->value += result->child_offset;
        cur_child->child_offset += result->child_offset;
        cur_child = next_child;
    }
    PHNode ***buffer = &ph->heap_item_handle;
    size_t merged_children = 0;
    printf("---\n");
    while ((merged_children + 2) <= ph->size) {
        (*buffer)[merged_children / 2] = _linking(
                (*buffer)[merged_children], (*buffer)[merged_children + 1]);
        merged_children += 2;
        printf("val2: %lf\n", (*buffer)[merged_children / 2]->value);
    }
    printf("---\n");
    if (merged_children != ph->size) {
        (*buffer)[merged_children / 2] = (*buffer)[merged_children];
        ph->size = merged_children / 2 + 1;
        printf("val3: %lf\n", (*buffer)[merged_children / 2]->value);
    } else {
        ph->size = merged_children / 2;
    }
    printf("number of nodes: %ld \n", ph->size);
    if (ph->size > 0) {
        ph->root = (*buffer)[ph->size - 1];
        for (size_t ii = ph->size - 2; ii >= 0; --ii) {
            printf("current root: value:%lf\n", ph->root->value);
            ph->root = _linking(ph->root, (*buffer)[ii]);
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


void ph_print_heap(PairingHeap *ph) {
    printf("------------ start -------------\n");
    if (hp_is_empty(ph)) {
        printf("hp is empty.\n");
    } else {
        printf("hp is not empty.\n");
    }
    printf("capacity of the heap: %ld\n", ph->capacity);
    printf("number of elements in this heap: %ld\n", ph->size);
    for (int i = 0; i < ph->size; i++) {
        printf("node[%d] value: %lf, payload: %d, offset: %lf \n", i,
               ph->heap_buffer[i].value, ph->heap_buffer[i].payload,
               ph->heap_buffer[i].child_offset);
    }
    printf("-------------------------------\n");
}


#endif //PROJ_C_PAIRING_HEAP_H
