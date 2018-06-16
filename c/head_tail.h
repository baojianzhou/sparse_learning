//
// Created by baojian on 6/15/18.
//

#ifndef PROJ_C_HEAD_TAIL_H
#define PROJ_C_HEAD_TAIL_H

#include <stdbool.h>
#include <limits.h>


void print_vector(int *vector, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n");
}

extern inline void init_bool(bool *vector, bool val, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = val;
    }
}

typedef struct Tuple {
    int node_ind;
    int edge_ind;
    bool is_visited;
} Tuple;

typedef struct Tour {
    int *nodes;
    int *edges;
    int len;
} Tour;

typedef struct WeiPair {
    double first;
    int index;
} WeiPair;
typedef struct Edge {
    int first;
    int second;
} Edge;

typedef struct Tree {
    int *nodes;
    int *edges;
    int num_nodes;
    int num_edges;
    double cost_t;
    double prize_t;
    int tmp_ind_edge;
    int tmp_ind_node;
} Tree;

typedef struct Forest {
    int *nodes;
    int *edges;
    int num_nodes;
    int num_edges;
    int num_cc; //connected components
    double cost_f;
    double prize_f;
    Tree **tree;
    int tmp_ind_edge;
    int tmp_ind_node;
} Forest;

typedef struct HeadProj {
    const Edge *edges;
    double *costs;
    double *prizes;
    const int g;
    const int s;
    const double C;
    const double delta;
    const int verbose_level;
    double total_prizes;
    int num_nodes;
    int num_edges;
    int *re_nodes;
    int *re_edges;
    int re_num_cc;
    int *tmp_nodes_map;
    bool *tmp_vector;
} HeadProj;

typedef struct TailProj {
    const Edge *edges;
    double *costs;
    double *prizes;
    const int g;
    const int s;
    const double C;
    const double nu;
    const double delta;
    int verbose_level;
    double total_prizes;
    int num_nodes;
    int num_edges;
    int *re_nodes;
    int *re_edges;
    int re_num_cc;
} TailProj;

typedef struct PCST {
    Edge *edges;
    double *costs;
    double *prizes;
} PCST;

typedef struct ProjApp {
    HeadProj *h_proj;
    TailProj *t_proj;
    PCST *pcst;
} ProjApp;


extern inline int comp_func(const void *e1, const void *e2) {
    if ((*(WeiPair *) e1).first > (*(WeiPair *) e2).first) return 1;
    if ((*(WeiPair *) e1).first < (*(WeiPair *) e2).first) return -1;
    return 0;
}


ProjApp *create_proj_app(Edge *edges, double *weights,
                         int s, int g, double budget,
                         int num_nodes, int num_edges) {
    ProjApp *app = (ProjApp *) malloc(sizeof(ProjApp));
    app->h_proj = (HeadProj *) malloc(sizeof(HeadProj));
    app->t_proj = (TailProj *) malloc(sizeof(TailProj));
    app->pcst = (PCST *) malloc(sizeof(PCST));
    app->h_proj->edges = edges;
    app->h_proj->costs = weights;
    app->h_proj->num_nodes = num_nodes;
    app->h_proj->num_edges = num_edges;
    app->h_proj->tmp_nodes_map = malloc(sizeof(int) * num_nodes);
    app->h_proj->tmp_vector = malloc(sizeof(bool) * num_nodes);
    return app;
}

bool free_proj_app(ProjApp *app) {
    free(app->h_proj);
    free(app->t_proj);
    free(app->pcst);
    free(app);
    return true;
}


double _min_pi(const double *prizes, int n) {
    double min_posi = 0.0;
    for (int i = 0; i < n; i++) {

    }
    return min_posi;
}

// calculate c(F)
double _cost_f() {
    return 0.0;
}

// calculate pi(i) for each node i
double *_pi_lambda(double *lambda_pi, double lambda, int n) {
    for (int i = 0; i < n; i++) {
        lambda_pi[i] *= lambda;
    }
}

// run fast-pcst algorithm.
Forest *_pcsf_gw(double *prizes) {
    return NULL;
}


/**
 * This method is to find a euler tour given a tree. This method is
 * proposed in the following paper:
 * Authors : Edmonds, Jack, and Ellis L. Johnson.
 * Title : Matching, Euler tours and the Chinese postman
 * Journal: Mathematical programming 5.1 (1973): 88-124.
 */
Tour *_dfs_tour(const HeadProj *h_proj, const Tree *tree) {
    if (tree->num_nodes <= 1) {
        printf("error: The input tree must have at least two nodes.");
        exit(0);
    }
    for (int i = 0; i < tree->num_nodes; i++) { // node to index map
        h_proj->tmp_nodes_map[tree->nodes[i]] = i;
    }
    int *nodes_map = h_proj->tmp_nodes_map;
    // adj: <original,neighbors>
    Tuple **adj = (Tuple **) malloc(tree->num_nodes * sizeof(Tuple *));
    size_t adj_size = (size_t) tree->num_nodes;
    int *adj_size_list = (int *) calloc((size_t) adj_size, sizeof(int)), i;
    for (i = 0; i < tree->num_edges; i++) {
        adj_size_list[nodes_map[h_proj->edges[tree->edges[i]].first]]++;
        adj_size_list[nodes_map[h_proj->edges[tree->edges[i]].second]]++;
    }
    for (i = 0; i < tree->num_nodes; i++) {
        int nei_size = adj_size_list[i];
        Tuple *neighbors = (Tuple *) malloc(sizeof(Tuple) * nei_size);
        adj[i] = neighbors;
    }
    int *tmp_inc_size = (int *) calloc((size_t) adj_size, sizeof(int));
    for (i = 0; i < tree->num_edges; i++) {
        int index_u = nodes_map[h_proj->edges[tree->edges[i]].first];
        int index_v = nodes_map[h_proj->edges[tree->edges[i]].second];
        // each tuple is: (indexed node, edge_index, is_visited)
        Tuple nei_u, nei_v;
        nei_u.node_ind = index_u;
        nei_u.edge_ind = tree->edges[i];
        nei_u.is_visited = false;
        nei_v.node_ind = index_v;
        nei_v.edge_ind = tree->edges[i];
        nei_v.is_visited = false;
        adj[index_u][tmp_inc_size[index_u]++] = nei_v;// edge u --> v
        adj[index_v][tmp_inc_size[index_v]++] = nei_u;// edge v --> u
    }
    int tour_nodes_size = 2 * tree->num_nodes - 1;
    int tour_edges_size = 2 * tree->num_nodes - 2;
    int *tour_nodes = (int *) malloc(tour_nodes_size * sizeof(int));
    int *tour_edges = (int *) malloc(tour_edges_size * sizeof(int));
    int start_node = nodes_map[tree->nodes[0]]; //the first element as root.
    int index_tour_nodes = 0, index_tour_edges = 0;
    bool *visited = (bool *) calloc((size_t) adj_size, sizeof(bool));
    tour_nodes[index_tour_nodes++] = start_node; // the first node
    while (true) {
        bool flag_1 = false;
        visited[start_node] = true;
        // iterate the neighbors of each node. in this loop, we check if
        // there exists any its neighbor which has not been visited.
        for (i = 0; i < adj_size_list[start_node]; i++) {
            int next_node = adj[start_node][i].node_ind;
            int edge_index = adj[start_node][i].edge_ind;
            if (!visited[next_node]) { // first time to visit this node.
                visited[next_node] = true; // mark it as labeled.
                tour_nodes[index_tour_nodes++] = next_node;
                tour_edges[index_tour_edges++] = edge_index;
                adj[start_node][i].is_visited = true; // mark it as labeled.
                start_node = next_node;
                flag_1 = true;
                break;
            }
        }
        // all neighbors are first-timely visited. Then we check if
        // there exists neighbors which is false nodes.
        if (!flag_1) {
            bool flag_2 = false;
            for (i = 0; i < adj_size_list[start_node]; i++) {
                int next_node = adj[start_node][i].node_ind;
                int edge_index = adj[start_node][i].edge_ind;
                bool is_visited = adj[start_node][i].is_visited;
                if (!is_visited) { // there exists a neighbor. has false node
                    adj[start_node][i].is_visited = true;
                    tour_nodes[index_tour_nodes++] = next_node;
                    tour_edges[index_tour_edges++] = edge_index;
                    start_node = next_node;
                    flag_2 = true;
                    break;
                }

            }
            // all nodes are visited and there is no false nodes.
            if (!flag_2) {
                break;
            }
        }
    }
    // convert indexed nodes back to original nodes.
    for (i = 0; i < tour_nodes_size; i++) {
        int original_node = tree->nodes[tour_nodes[i]];
        tour_nodes[i] = original_node;
    }
    Tour *tour = (Tour *) malloc(sizeof(Tour));
    tour->nodes = tour_nodes;
    tour->edges = tour_edges;
    tour->len = tour_nodes_size;
    // to free all used memory.
    free(tmp_inc_size);
    free(visited);
    free(adj_size_list);
    for (i = 0; i < tree->num_nodes; i++) {
        free(adj[i]);
    }
    free(adj);
    free(nodes_map);
    return tour;
}


bool _prune_tree(HeadProj *h_proj, Tree *t, double c_prime) {
    int i, j, k;
    Tour *tour = _dfs_tour(h_proj, t);
    // calculating pi_prime.
    double *pi_prime = (double *) malloc(sizeof(double) * tour->len);
    init_bool(h_proj->tmp_vector, true, h_proj->num_nodes);
    for (j = 0; j < tour->len; j++) {
        int v_j = tour->nodes[j];
        if (h_proj->tmp_vector[v_j]) { // first time show in the tour.
            pi_prime[j] = h_proj->prizes[v_j];
            h_proj->tmp_vector[v_j] = false;
        } else {
            pi_prime[j] = 0.0;
        }
    }
    double phi = t->prize_t / t->cost_t;
    for (i = 0; i < tour->len; i++) {
        int node = tour->nodes[i];
        if (h_proj->prizes[node] >= ((c_prime * phi) / 6.)) {
            t->nodes[0] = node; // create a single node tree.
            t->edges = NULL;
            t->num_nodes = 1;
            t->num_edges = 0;
            t->prize_t = h_proj->prizes[node];
            t->cost_t = 0.0;
            free(pi_prime);
            return true;
        }
    }
    int *p_l = (int *) malloc(sizeof(int) * tour->len), p_l_size = 0;
    for (i = 0; i < tour->len; i++) {
        p_l[i] = i, p_l_size++; // add element
        double pi_prime_pl = 0.0;
        for (j = 0; j < p_l_size; j++) {
            pi_prime_pl += pi_prime[j];
        }
        double c_prime_pl = 0.0;
        if (p_l_size >= 2) { // <= 1: there is no edge.
            for (k = 0; k < p_l_size - 1; k++) {
                c_prime_pl += h_proj->costs[tour->edges[p_l[k]]];
            }
        }
        if (c_prime_pl > c_prime) { // start a new sublist
            p_l_size = 0;
        } else if (pi_prime_pl >= ((c_prime * phi) / 6.)) {
            t->num_nodes = p_l_size; // create a new subtree.
            t->num_edges = p_l_size;
            t->cost_t = 0.0;
            t->prize_t = 0.0;
            for (int ind = 0; ind < p_l_size; ind++) {
                int index = p_l[ind];
                t->nodes[ind] = tour->nodes[index];
                t->edges[ind] = tour->edges[index];
                t->cost_t += h_proj->costs[t->nodes[ind]];
                t->prize_t += h_proj->prizes[t->nodes[ind]];
            }
            free(p_l);
            free(pi_prime);
            return true;
        }
    }
    printf("Error Never reach at this point.");
    exit(0);
}


WeiPair *_sort_forest(const HeadProj *h_proj, Forest *f) {
    // Create a descending ordered forest.
    int i;
    for (i = 0; i < f->num_nodes; i++) {
        h_proj->tmp_nodes_map[f->nodes[i]] = i;
    }
    int *nodes_map = h_proj->tmp_nodes_map;
    // Create a adj graph. Node ID starts from 0.
    int **adj = (int **) malloc(sizeof(int *) * f->num_nodes);
    size_t adj_size = (size_t) f->num_nodes;
    int *adj_size_list = (int *) calloc((size_t) adj_size, sizeof(int));
    for (int index = 0; index < f->num_edges; index++) {
        int index_u = nodes_map[h_proj->edges[f->edges[index]].first];
        int index_v = nodes_map[h_proj->edges[f->edges[index]].second];
        adj_size_list[index_u]++;
        adj_size_list[index_v]++;
    }
    for (size_t index = 0; index < f->num_nodes; index++) {
        adj[index] = (int *) malloc(sizeof(int) * adj_size_list[index]);
    }
    int t = 0; // component id
    int *tree_size_list = (int *) calloc((size_t) f->num_cc, sizeof(int));
    //label the nodes to the components id.
    int *comp = (int *) calloc(adj_size, sizeof(int));
    bool *visited = (bool *) calloc(adj_size, sizeof(bool));
    int *stack = (int *) malloc(adj_size * sizeof(int)); // create a stack
    for (i = 0; i < adj_size; i++) { // dfs algorithm to get cc
        if (!visited[i]) {
            int top = 0;// let stack be empty
            stack[top++] = i; // push i
            while (top != 0) {
                int s = stack[(top--) - 1];// pop
                if (!visited[s]) {
                    visited[s] = true;
                    comp[s] = t;
                    tree_size_list[t] += 1;
                }
                for (int k = 0; k < adj_size_list[s]; k++) {
                    if (!visited[k]) {
                        stack[top++] = k;
                    }
                }
            }
            t++; // to label component id.
        }
    }
    int num_cc = t; //the total number of connected components in the forest.
    // A collection of trees.
    Tree **trees = (Tree **) malloc(num_cc * sizeof(Tree *));
    for (i = 0; i < num_cc; i++) { // initialize trees.
        trees[i] = (Tree *) malloc(sizeof(Tree));
        trees[i]->num_nodes = tree_size_list[i];
        trees[i]->num_edges = tree_size_list[i] - 1;
        trees[i]->prize_t = 0.0;
        trees[i]->cost_t = 0.0;
        trees[i]->tmp_ind_edge = 0;
        trees[i]->tmp_ind_node = 0;
        trees[i]->nodes = (int *) malloc(sizeof(int) * trees[i]->num_nodes);
        trees[i]->edges = (int *) malloc(sizeof(int) * trees[i]->num_edges);
    }
    // nodes in each tree.
    for (i = 0; i < f->num_nodes; i++) {
        int tree_index = comp[i]; // construct each tree
        int node_i = f->nodes[i];
        trees[tree_index]->prize_t += h_proj->prizes[node_i];
        trees[tree_index]->nodes[trees[tree_index]->tmp_ind_node++] = node_i;
    }
    // try to insert edges into trees.
    for (i = 0; i < f->num_edges; i++) {
        int edge_index = f->edges[i];
        int u = h_proj->edges[edge_index].first; // select one endpoint
        int index_u = nodes_map[u];
        trees[comp[index_u]]->edges[trees[i]->tmp_ind_edge++] = edge_index;
        trees[comp[index_u]]->cost_t += h_proj->costs[edge_index];
    }

    // weights for sorting.
    WeiPair *trees_wei = (WeiPair *) malloc(num_cc * sizeof(WeiPair));
    for (i = 0; i < num_cc; i++) {
        double weight;
        if (trees[i]->cost_t > 0.0) {
            weight = trees[i]->prize_t / trees[i]->cost_t;
        } else {
            weight = trees[i]->prize_t / 1.; // for a single node tree
        }
        WeiPair wei;;
        wei.first = weight, wei.index = i;
        trees_wei[i] = wei;
    }
    f->tree = trees;
    qsort(trees_wei, (size_t) num_cc, sizeof(WeiPair), comp_func);
    return trees_wei;
}


// prune a forest
bool _prune_forest(HeadProj *h_proj, Forest *f) {
    int i, j;
    // case 1: usually, there is only one tree. to make it faster.
    if (h_proj->g == 1) {
        if (f->cost_f <= h_proj->C) {
        } else if (0.0 < h_proj->C) {
            _prune_tree(h_proj, f->tree[0], h_proj->C);
        } else {
            // A single node tree.
            int max_node = f->nodes[0];
            double max_prize = h_proj->prizes[max_node];
            for (i = 0; i < f->num_nodes; i++) {
                if (max_prize < h_proj->prizes[f->nodes[i]]) {
                    max_prize = h_proj->prizes[f->nodes[i]];
                    max_node = f->nodes[i];
                }
            }
            f->num_edges = 0;
            f->num_nodes = 1;
            f->nodes[0] = max_node;
            f->tmp_ind_node = 0;
            f->tmp_ind_edge = 0;
        }
        return true;
    }
    // case 2: there are at least two trees.
    WeiPair *sorted_f = _sort_forest(h_proj, f);
    double c_r = h_proj->C;
    for (i = 0; i < f->num_cc; i++) {
        Tree *cur_tree = f->tree[sorted_f[i].index];
        double c_tree_i = cur_tree->cost_t;
        if (c_r >= c_tree_i) { // do not updated tree
            c_r = c_r - c_tree_i;
        } else if (c_r > 0.0) {
            // tree_i must have at least two nodes and one edge.
            _prune_tree(h_proj, cur_tree, c_r);
            c_r = 0.0;
        } else {
            // A single node tree.
            int max_node = cur_tree->nodes[0];
            double max_prize = h_proj->prizes[max_node];
            for (i = 0; i < cur_tree->num_nodes; i++) {
                if (max_prize < h_proj->prizes[cur_tree->nodes[i]]) {
                    max_prize = h_proj->prizes[cur_tree->nodes[i]];
                    max_node = cur_tree->nodes[i];
                }
            }
            cur_tree->num_edges = 0;
            cur_tree->num_nodes = 1;
            cur_tree->nodes[0] = max_node;
            cur_tree->tmp_ind_node = 0;
            cur_tree->tmp_ind_edge = 0;
        }
    }// iterate trees by descending order.
    f->num_nodes = 0;
    f->num_edges = 0;
    f->prize_f = 0.0;
    f->cost_f = 0.0;
    f->tmp_ind_node = 0;
    f->tmp_ind_edge = 0;
    for (i = 0; i < f->num_cc; i++) {
        f->num_nodes += f->tree[i]->num_nodes;
        f->tree[i]->tmp_ind_node = 0;
        f->tree[i]->tmp_ind_edge = 0;
        for (j = 0; j < f->tree[i]->num_nodes; j++) {
            f->nodes[f->tmp_ind_node++] =
                    f->tree[i]->nodes[f->tree[i]->tmp_ind_node++];
        }
        for (j = 0; j < f->tree[i]->num_edges; j++) {
            f->edges[f->tmp_ind_edge++] =
                    f->tree[i]->edges[f->tree[i]->tmp_ind_edge++];
        }
        f->num_edges += f->tree[i]->num_edges;
    }
}


Forest *tail_run(Edge *edges, double *costs, double *prizes,
                 int g, int s, double C, double nu, double delta) {
    return NULL;
}

Forest *head_run(HeadProj *h_proj) {
    // TODO make sure prizes must has at least one non-zero entry.
    int i = 0;
    double pi_min = LONG_MAX, lambda_r, lambda_l, epsilon;
    Forest *f;
    h_proj->total_prizes = 0.0; // total_prizes will be calculated.
    for (i = 0; i < h_proj->num_nodes; i++) {
        h_proj->total_prizes += h_proj->prizes[i];
        if ((h_proj->prizes[i] < pi_min) &&
            (h_proj->prizes[i] > 0.0)) {
            pi_min = h_proj->prizes[i];
        }
    }
    /**
     *Warning: There is a precision issue here. We may need to define a
     * minimum precision. Since, in our experiment,  we found that some very
     * small positive number could be like 1.54046e-310, 1.54046e-310.
     * In this case, the fast-pcst cannot stop!!!
     */
    if (pi_min < 1e-8) {
        if (h_proj->verbose_level > 0) {
            printf("warning too small positive value found ");
        }
        pi_min = 1e-8;
    }
    lambda_r = (2. * h_proj->C) / pi_min;
    lambda_l = 1. / (4. * h_proj->total_prizes);
    epsilon =
            (h_proj->delta * h_proj->C) / (2. * h_proj->total_prizes);
    f = _pcsf_gw(
            _pi_lambda(h_proj->prizes, lambda_r, h_proj->num_nodes));
    // ensure that we have invariant c(F_r) > 2 C
    if (f->cost_f <= 2. * h_proj->C) {
        return f;
    }
    while ((lambda_r - lambda_l) > epsilon) {
        double lambda_m = (lambda_l + lambda_r) / 2.;
        if (h_proj->verbose_level > 0) {
            printf(" lambda_r: %lf, lambda_l: %lf", lambda_r,
                   lambda_l);
            printf(" lambda_m: %lf, gap: %lf", lambda_m,
                   ((lambda_r - lambda_l)));
        }
        f = _pcsf_gw(
                _pi_lambda(h_proj->prizes, lambda_m,
                           h_proj->num_nodes));
        if (f->cost_f > (2. * h_proj->C)) {
            lambda_r = lambda_m;
        } else {
            lambda_l = lambda_m;
        }
    } // binary search over the Lagrange parameter lambda
    Forest *f_l = _pcsf_gw(
            _pi_lambda(h_proj->prizes, lambda_l, h_proj->num_nodes));
    Forest *f_r = _pcsf_gw(
            _pi_lambda(h_proj->prizes, lambda_r, h_proj->num_nodes));
    // It seems that this method is not necessary.
    _prune_forest(h_proj, f_r);
    if (f_l->prize_f >= f_r->prize_f) {
        return f_l;
    } else {
        return f_r;
    }
}

#endif //PROJ_C_HEAD_TAIL_H
