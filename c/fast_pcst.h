/*****************************************************************************
 *
 * Created by baojian on 6/3/18.
 *
*****************************************************************************/

#ifndef PROJ_C_FAST_PCST_H
#define PROJ_C_FAST_PCST_H


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "pairing_heap.h"

/****************************************************************************/

typedef struct PQueueNode { // Priority Queue Node
    int priority;
    double data;
    struct PQueueNode *next;
} PQueueNode;
typedef struct PQueue { //Priority Queue
    int active_nodes;
    int minimum_pages;
    PQueueNode *nodes;
} PQueue;

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
    PairingHeap *heap_node;
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
    PHNode *edge_parts;
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
    PHNode *pairing_heap_buffer;
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

        if (pcst->clusters[ii].active) { ;
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

#endif //PROJ_C_FAST_PCST_H
