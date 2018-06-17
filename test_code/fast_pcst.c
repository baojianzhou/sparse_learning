
/*****************************************************************************
 * This C program implements the data structure, pairing_heap.
 * Please check details in the following paper:
 * [1]  Fredman, M.L., Sedgewick, R., Sleator, D.D. and Tarjan, R.E., 1986.
 *      The pairing heap: A new form of self-adjusting heap. Algorithmica,
 *      1(1-4), pp.111-129.
 * Created by baojian on 6/3/18.
 *
*****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

/****************************************************************************/

typedef double key_type; // a real-valued type.
typedef double data_type; // should be comparable.
typedef struct PQueueNode { // Priority Queue Node
    int priority;
    data_type data;
    struct PQueueNode *next;
} PQueueNode;
typedef struct PQueue { //Priority Queue
    int active_nodes;
    int minimum_pages;
    PQueueNode *nodes;
} PQueue;
typedef struct PHeapNode { // Pairing Heap Node
    key_type key;
    data_type data;
    struct PHeapNode *child; // its first child
    struct PHeapNode *parent; // to its parent in this binary tree
    struct PHeapNode *sibling; // point to its next older sibling
} PHeapNode;
typedef enum PruningMethod {
    None, Simple, GW, Strong
} PruningMethod;
typedef struct EdgePair {
    int first;
    int second;
} EdgePair;
typedef struct KeyPair {
    int first;
    double second;
} KeyPair;
typedef struct BoolIntPair {
    bool first;
    int second;
} BoolIntPair;
typedef struct EdgePart {
    double next_event_val;
    bool deleted;
    PHeapNode *heap_node;
} EdgePart;
typedef struct EdgeInfo {
    int inactive_merge_event;
} EdgeInfo;
typedef struct InactiveMergeEvent {
    int active_cluster_index;
    int inactive_cluster_index;
    int active_cluster_node;
    int inactive_cluster_node;
} InactiveMergeEvent;
typedef struct Cluster {
    PHeapNode *edge_parts;
    bool active;
    double active_start_time;
    double active_end_time;
    int merged_into;
    double prize_sum;
    double subcluster_moat_sum;
    double moat;
    bool contains_root;
    int skip_up;
    double skip_up_sum;
    int merged_along;
    int child_cluster_1;
    int child_cluster_2;
    bool necessary;
} Cluster;
typedef struct FastPcst {
    EdgePair *edges;
    double *prizes;
    double *costs;
    int root;
    int num_trees;
    int n; // total n nodes in graph
    int m; // total m undirected edges in graph
    PruningMethod *pruning;
    int verbosity_level;
    PHeapNode *pairing_heap_buffer;
    EdgePart *edge_parts;
    EdgeInfo *edge_info;
    Cluster *clusters;
    InactiveMergeEvent *inactive_merge_event;
    PQueueNode *clusters_deactivation;
    PQueueNode clusters_next_edge_event;
    double current_time;
    double eps;
    bool *node_good;
    bool *node_deleted;
    int *phase2_result;
    KeyPair *path_compression_visited;
    int *cluster_queue;
    KeyPair **phase3_neighbors;
    int *final_component_label;
    int **final_components;
    int root_component_index;
    KeyPair *strong_pruning_parent;
    double *strong_pruning_payoff;
    BoolIntPair *stack;
    int *stack2;

} FastPcst;

int pcst_fast(EdgePair *edges, double *prizes, double *cost, int root,
              int num_trees, PruningMethod *pruning, int verbose_level,
              int n, int m);

int pcst_fast(EdgePair *edges, double *prizes, double *costs, int root,
              int num_trees, PruningMethod *pruning, int verbose_level,
              int n, int m) {
    FastPcst *pcst = (FastPcst *) malloc(sizeof(FastPcst));
    pcst->edges = edges;
    pcst->prizes = prizes;
    pcst->costs = costs;
    pcst->root = root;
    pcst->num_trees = num_trees;
    pcst->pruning = pruning;
    pcst->verbosity_level = verbose_level;
    // initialize
    pcst->edge_parts = (EdgePart *) malloc(sizeof(EdgePart) * (2 * m));
    pcst->node_deleted = (bool *) calloc(sizeof(bool) * n, sizeof(bool));
    pcst->edge_info = (EdgeInfo *) malloc(sizeof(EdgeInfo) * m);
    for (int ii = 0; ii < m; ii++) {
        pcst->edge_info[ii].inactive_merge_event = -1;
    }
    pcst->clusters = (Cluster *) malloc(sizeof(Cluster) * n);
    pcst->current_time = 0.0;
    pcst->eps = 1e-10; //TODO need to change it further.
    for (int ii = 0; ii < n; ii++) {
        pcst->clusters[ii].edge_parts = pcst->pairing_heap_buffer;
        pcst->clusters[ii].active = (ii != root);
        pcst->clusters[ii].active_start_time = 0.0;
        pcst->clusters[ii].active_end_time = -1.0;
        if (ii == root) {
            pcst->clusters[ii].active_end_time = 0.0;
        }
        pcst->clusters[ii].merged_into = -1;
        pcst->clusters[ii].prize_sum = prizes[ii];
        pcst->clusters[ii].subcluster_moat_sum = 0.0;
        pcst->clusters[ii].moat = 0.0;
        pcst->clusters[ii].contains_root = (ii == root);
        pcst->clusters[ii].skip_up = -1;
        pcst->clusters[ii].skip_up_sum = 0.0;
        pcst->clusters[ii].merged_along = -1;
        pcst->clusters[ii].child_cluster_1 = -1;
        pcst->clusters[ii].child_cluster_2 = -1;
        pcst->clusters[ii].necessary = false;

        if (pcst->clusters[ii].active) {
            ;
        }
    }

    // last stage is to free the memory
    free(pcst->edge_info);
    free(pcst->node_deleted);
    free(pcst->edge_parts);
    free(pcst);
}

/****************************************************************************/
bool pq_create() {

}

bool pq_get_min(PQueueNode *q) {
    if (q == NULL) {
        return false;
    }
    PQueueNode *min_node = (PQueueNode *) malloc(sizeof(PQueueNode));
    min_node->data;
    return min_node;
}

bool pq_delete_min() {
    return true;
}

bool pq_insert() {
    return true;
}

void pq_decrease_key() {
}

void pq_delete_element() {

}


/****************************************************************************/

PHeapNode *ph_make_heap();

data_type ph_find_min(PHeapNode *h);

PHeapNode *ph_insert(data_type data_x, PHeapNode *h);

PHeapNode *ph_delete_min(PHeapNode *h);

PHeapNode *ph_meld(PHeapNode *h1, PHeapNode *h2);

PHeapNode *ph_decrease_key();

bool *ph_delete(data_type data_x, PHeapNode *h);

PHeapNode *_merge_pairs();

PHeapNode *_linking(PHeapNode *t1, PHeapNode *t2);

/** create a new, empty heap named h */
PHeapNode *ph_make_heap() {
    PHeapNode *h = NULL;
    return h;
}

/** return an item of minimum key from heap h, without changing h*/
data_type ph_find_min(PHeapNode *h) {
    if (h == NULL) {
        perror("pairing heap h is empty !");
        exit(0);
    }
    return h->data;
}


/** insert item data_x, with predefined key, into heap h,
 * not previously containing data_x*/
PHeapNode *ph_insert(data_type data_x, PHeapNode *h) {
    PHeapNode *new_node = (PHeapNode *) malloc(sizeof(PHeapNode));
    new_node->data = data_x;
    new_node->child = NULL;
    new_node->parent = NULL;
    new_node->sibling = NULL;
    return _linking(new_node, h);
}

/** delete an item of minimum key from heap h and return it.
 * If h is originally empty, return a special null item.*/
PHeapNode *ph_delete_min(PHeapNode *h) {
    if (h == NULL) {
        perror("pairing heap PHeapNode is empty !");
        return NULL;
    } else {
        return _merge_pairs();
    }
}

/** return the heap formed by taking the union of the item-disjoint
 * heaps h1 and h2. Melding destroys h1 and h2.*/
PHeapNode *ph_meld(PHeapNode *h1, PHeapNode *h2) {
    if (h1 == NULL) {
        return h2;
    } else if (h2 == NULL) {
        return h1;
    } else if (h1->data < h2->data) {
        return NULL;
    } else {
        return h1;
    }
}

/** decrease the key of item x in heap h by subtracting the non-negative
 * real number delta*/
PHeapNode *ph_decrease_key() {

}

/** delete item x from heap h, known to contain it.*/
bool *ph_delete(data_type data_x, PHeapNode *h) {
    return true;
}


/** merge-pairs*/
PHeapNode *_merge_pairs() {
    return NULL;
}

// use linking to combine two heap-ordered trees
PHeapNode *_linking(PHeapNode *t1, PHeapNode *t2) {
    if (t1 == NULL) {
        return t2;
    } else if (t2 == NULL) {
        return t1;
    } else {
        PHeapNode *small_node = t1;
        PHeapNode *large_node = t2;
        if (t1->data > t2->data) {
            small_node = t2;
            large_node = t1;
        }
        // first child of small node
        large_node->sibling = small_node->child;
        if (large_node->sibling != NULL) {
            large_node->sibling->parent = large_node;
        }
        large_node->parent = small_node;
        small_node->child = large_node;
        large_node->data -= 0.; // TODO
        large_node->child;
        return small_node;
    }
}

/****************************************************************************/
int main(int argc, char *argv[]) {
    EdgePair *p = (EdgePair *) malloc(sizeof(EdgePair));
    p->first = 1;
    p->second = 2;
    printf("%d %d\n", p->first, p->second);
    bool *x = (bool *) calloc(sizeof(bool) * 10, sizeof(bool));
    int i = 0;
    for (i = 0; i < 10; i++) {
        printf("%d\n", x[i]);
    }
}