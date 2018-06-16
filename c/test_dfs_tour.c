#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "head_tail.h"


typedef struct Grid {
    Edge **edges;
    double *weights;
    int num_edges;
    int num_nodes;
} Grid;

Grid *get_grid_graph(int num_nodes) {
    // make them empty.
    int num_edges = (4 * num_nodes) + (num_nodes - 2);
    Edge **edges = (Edge **) malloc(sizeof(Edge *) * num_edges);
    double *weights = (double *) malloc(sizeof(double) * num_edges);
    int index = 0;
    auto length = (int) sqrt(num_nodes);
    int edge_index = 0;
    // generate a graph which is a grid graph.
    int width = length;
    Edge *edge;
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < width; j++) {
            if ((index % length) != (length - 1)) {
                edge = malloc(sizeof(Edge));
                edge->first = index;
                edge->second = index + 1;
                edges[edge_index++] = edge;
                if ((index + length) < num_nodes) {
                    edge = malloc(sizeof(Edge));
                    edge->first = index;
                    edge->second = index + length;
                    edges[edge_index++] = edge;
                }
            } else {
                if ((index + length) < num_nodes) {
                    edge = malloc(sizeof(Edge));
                    edge->first = index;
                    edge->second = index + length;
                    edges[edge_index++] = edge;
                }
            }
            index += 1;
        }
    }
    for (size_t i = 0; i < edge_index; i++) {
        weights[i] = 1.0;
    }
    Grid *grid = (Grid *) malloc(sizeof(Grid));
    grid->num_nodes = num_nodes;
    grid->edges = edges;
    grid->weights = weights;
    return grid;
}

bool test_case_1() {
    int num_nodes = 11;
    int num_edges = 4;
    Edge *edges = (Edge *) malloc(sizeof(Edge) * num_edges);
    int edge_i[] = {6, 7, 8, 9};
    int edge_j[] = {7, 8, 9, 10};
    for (int i = 0; i < num_edges; i++) {
        edges[i].first = edge_i[i];
        edges[i].second = edge_j[i];
    }
    // construct a testing tree
    Tree *tree = (Tree *) malloc(sizeof(Tree));
    tree->nodes = (int *) malloc(sizeof(int) * 5);
    tree->nodes[0] = 6;
    tree->nodes[1] = 7;
    tree->nodes[2] = 8;
    tree->nodes[3] = 9;
    tree->nodes[4] = 10;
    tree->edges = (int *) malloc(sizeof(int) * 4);
    tree->edges[0] = 0;
    tree->edges[1] = 1;
    tree->edges[2] = 2;
    tree->edges[3] = 3;
    tree->num_nodes = 5;
    tree->num_edges = 4;
    tree->cost_t = 0.0;
    tree->prize_t = 0.0;
    ProjApp *app = create_proj_app(edges, NULL, 0, 0, 0.0, num_nodes,
                                   num_edges);
    Tour *tour = _dfs_tour(app->h_proj, tree);
    print_vector(tour->nodes, tour->len);
    print_vector(tour->edges, tour->len - 1);
    free_proj_app(app);
    printf("finish the test.\n");
    return true;
}

bool test_case_2() {
    int num_nodes = 11;
    int num_edges = 5;
    Edge *edges = (Edge *) malloc(sizeof(Edge) * 5);
    int edge_i[] = {1, 1, 1, 3, 5};
    int edge_j[] = {2, 3, 4, 5, 6};
    for (int i = 0; i < 5; i++) {
        edges[i].first = edge_i[i];
        edges[i].second = edge_j[i];
    }
    // construct a testing tree
    Tree *tree = (Tree *) malloc(sizeof(Tree));
    tree->nodes = (int *) malloc(sizeof(int) * 6);
    tree->nodes[0] = 1;
    tree->nodes[1] = 2;
    tree->nodes[2] = 3;
    tree->nodes[3] = 4;
    tree->nodes[4] = 5;
    tree->nodes[5] = 6;
    tree->edges = (int *) malloc(sizeof(int) * 5);
    tree->edges[0] = 0;
    tree->edges[1] = 1;
    tree->edges[2] = 2;
    tree->edges[3] = 3;
    tree->edges[4] = 4;
    tree->cost_t = 0.0;
    tree->prize_t = 0.0;
    tree->num_nodes = 6;
    tree->num_edges = 5;
    ProjApp *app = create_proj_app(edges, NULL, 0, 0, 0.0, num_nodes,
                                   num_edges);
    Tour *tour = _dfs_tour(app->h_proj, tree);
    print_vector(tour->nodes, tour->len);
    print_vector(tour->edges, tour->len - 1);
    free_proj_app(app);
    printf("finish the test.\n");
    return true;
}


bool test_case_3() {
    Tree *tree = (Tree *) malloc(sizeof(Tree));
    int nodes_t[] = {1, 2, 3, 4, 5};
    int edges_t[] = {0, 1, 2, 3};
    tree->nodes = nodes_t;
    tree->edges = edges_t;
    tree->num_nodes = 5;
    tree->num_edges = 4;
    tree->cost_t = 0.0;
    tree->prize_t = 0.0;
    Edge *edges = (Edge *) malloc(sizeof(Edge) * 5);
    double weights[] = {1., 1., 1., 1., 1., 1., 1., 1.};
    int edge_i[] = {1, 1, 1, 1};
    int edge_j[] = {2, 3, 4, 5};
    int s = 3, g = 1, i = 0;
    double budget = 1.0;
    for (i = 0; i < 4; i++) {
        edges[i].first = edge_i[i];
        edges[i].second = edge_j[i];
    }
    int num_nodes = 11;
    int num_edges = 4;
    ProjApp *app = create_proj_app(edges, weights, s, g, budget,
                                   num_nodes, num_edges);
    Tour *tour = _dfs_tour(app->h_proj, tree);
    print_vector(tour->nodes, tour->len);
    print_vector(tour->edges, tour->len - 1);
    free_proj_app(app);
    printf("finish the test.\n");
    return true;
}


bool test_case_4() {
    Tree *tree = (Tree *) malloc(sizeof(Tree));
    int nodes_t[] = {3, 4, 5, 6, 7, 9, 8, 15, 16, 12, 11, 10, 13, 14};
    int edges_t[] = {8, 7, 6, 5, 4, 1, 2, 3, 0, 11, 13, 15, 17};
    tree->nodes = nodes_t;
    tree->edges = edges_t;
    tree->num_nodes = 14;
    tree->num_edges = 13;
    tree->cost_t = 0.0;
    tree->prize_t = 0.0;
    Edge *edges = (Edge *) malloc(sizeof(Edge) * 18);
    double weights[] = {1., 1., 1., 1., 1., 1., 1., 1.};
    // some fake edges are added.
    int edge_i[] = {11, 11, 12, 12, 5, 5, 4, 4, 3, -1,
                    -1, 15, -1, 8, -1, 7, -1, 7};
    int edge_j[] = {10, 12, 13, 14, 11, 7, 6, 5, 4, -1,
                    -1, 16, -1, 15, -1, 8, -1, 9};
    int s = 3, g = 1, i = 0;
    double budget = 1.0;
    for (i = 0; i < 18; i++) {
        edges[i].first = edge_i[i];
        edges[i].second = edge_j[i];
    }
    int num_nodes = 20;
    int num_edges = 18;
    ProjApp *app = create_proj_app(edges, weights, s, g, budget,
                                   num_nodes, num_edges);
    Tour *tour = _dfs_tour(app->h_proj, tree);
    print_vector(tour->nodes, tour->len);
    print_vector(tour->edges, tour->len - 1);
    free_proj_app(app);
    printf("finish the test.\n");
    return true;
}

int main() {
    test_case_1();
    test_case_2();
    test_case_3();
    test_case_4();
    return 0;
}


/**
 * This method is to find a euler tour given a tree. This method is
 * proposed in the following paper:
 * Authors : Edmonds, Jack, and Ellis L. Johnson.
 * Title : Matching, Euler tours and the Chinese postman
 * Journal: Mathematical programming 5.1 (1973): 88-124.
 */
Tour *_dfs_tour_passed(const HeadProj *h_proj, const Tree *tree) {
    //Make sure the tree has at least two nodes.
    if (tree->num_nodes <= 1) {
        printf("error: The input tree has at least two nodes.");
        exit(0);
    }
    // node to index map
    for (int i = 0; i < tree->num_nodes; i++) {
        h_proj->tmp_nodes_map[tree->nodes[i]] = i;
    }
    int *nodes_map = h_proj->tmp_nodes_map;
    printf("mapped nodes \n");
    print_vector(nodes_map, h_proj->num_nodes);
    // adj: <original,neighbors>
    Tuple **adj = (Tuple **) malloc(tree->num_nodes * sizeof(Tuple *));
    size_t adj_size = (size_t) tree->num_nodes;
    int *adj_size_list = (int *) calloc((size_t) adj_size, sizeof(int)), i;
    for (i = 0; i < tree->num_edges; i++) {
        adj_size_list[nodes_map[h_proj->edges[tree->edges[i]].first]]++;
        adj_size_list[nodes_map[h_proj->edges[tree->edges[i]].second]]++;
    }
    printf("number of neighbors: \n");
    for (i = 0; i < tree->num_nodes; i++) {
        int nei_size = adj_size_list[i];
        Tuple *neighbors = (Tuple *) malloc(sizeof(Tuple) * nei_size);
        adj[i] = neighbors;
        printf("%d ", adj_size_list[i]);
    }
    printf("\nall edges: \n");
    int *tmp_inc_size = (int *) calloc((size_t) adj_size, sizeof(int));
    for (i = 0; i < tree->num_edges; i++) {
        int index_u = nodes_map[h_proj->edges[tree->edges[i]].first];
        int index_v = nodes_map[h_proj->edges[tree->edges[i]].second];
        printf("mapped edge (%d,%d) original edge (%d,%d)\n",
               index_u, index_v, h_proj->edges[tree->edges[i]].first,
               h_proj->edges[tree->edges[i]].second);
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

    for (i = 0; i < tree->num_nodes; i++) {
        printf("current node: %d, nei: ", i);
        for (int j = 0; j < adj_size_list[i]; j++) {
            printf("%d ", adj[i][j].node_ind);
        }
        printf("\n");
    }

    int tour_nodes_size = 2 * tree->num_nodes - 1;
    int tour_edges_size = 2 * tree->num_nodes - 2;
    int *tour_nodes = (int *) malloc(tour_nodes_size * sizeof(int));
    int *tour_edges = (int *) malloc(tour_edges_size * sizeof(int));
    //the first element as root.
    int start_node = nodes_map[tree->nodes[0]];
    int index_tour_nodes = 0, index_tour_edges = 0;
    bool *visited = (bool *) calloc((size_t) adj_size, sizeof(bool));
    tour_nodes[index_tour_nodes++] = start_node; // the first node
    printf("\ncurrent node_i: %d\n", tour_nodes[index_tour_nodes - 1]);
    while (true) {
        bool flag_1 = false;
        visited[start_node] = true;
        // iterate the neighbors of each node. in this loop, we check if
        // there exists any its neighbor which has not been visited.
        printf("current considered: %d, it has %d neis\n", start_node,
               adj_size_list[start_node]);
        for (i = 0; i < adj_size_list[start_node]; i++) {
            int next_node = adj[start_node][i].node_ind;
            int edge_index = adj[start_node][i].edge_ind;
            printf("next considered nei: %d\n", next_node);
            if (!visited[next_node]) { // first time to visit this node.
                visited[next_node] = true; // mark it as labeled.
                tour_nodes[index_tour_nodes++] = next_node;
                printf("first time visit node_i: %d\n", next_node);
                tour_edges[index_tour_edges++] = edge_index;
                adj[start_node][i].is_visited = true; // mark it as labeled.
                start_node = next_node;
                flag_1 = true;
                break;
            }
        }
        printf("--\n");
        // all neighbors are first-timely visited. Then we check if
        // there exists neighbors which is false nodes.
        if (!flag_1) {
            bool flag_2 = false;
            for (i = 0; i < adj_size_list[start_node]; i++) {
                int next_node = adj[start_node][i].node_ind;
                int edge_index = adj[start_node][i].edge_ind;
                bool is_visited = adj[start_node][i].is_visited;
                printf("next considered: %d\n", next_node);
                if (!is_visited) { // there exists a neighbor. has false node
                    adj[start_node][i].is_visited = true;
                    tour_nodes[index_tour_nodes++] = next_node;
                    printf("false node current node_i: %d\n",
                           tour_nodes[index_tour_nodes - 1]);
                    tour_edges[index_tour_edges++] = edge_index;
                    start_node = next_node;
                    flag_2 = true;
                    break;
                }

            }
            printf("----\n");
            if (!flag_2) {
                printf("finish the tour\n");
                break;// all nodes are visited and there is no false nodes.
            }
        }
    }
    if (index_tour_nodes != tour_nodes_size ||
        index_tour_edges != tour_edges_size) {
        printf("\n %d", index_tour_nodes);
        printf("\n %d", tour_nodes_size);
        printf("\n %d", index_tour_edges);
        printf("\n %d", tour_edges_size);
        printf("error! ");
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
    return tour; //original nodes, edge indices
}

