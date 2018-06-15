/*=========================================================
 * This main.cpp contains a Python wrapper. We use
 * Boost.Python wrapper. Created by
 *              Name : Baojian Zhou,
 *              Email: bzhou6@albany.edu
 *              Date : 05/30/17.
=========================================================*/
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <algorithm>
#include "pcst_fast.h"

using std::vector;
using std::pair;
using std::cout;
using std::endl;
using std::tuple;
using std::get;
using std::make_tuple;
using std::make_pair;
using std::numeric_limits;
using cluster_approx::PCSTFast;


int main_test() {
    vector<bool> test(10, false);
    for (auto i:test) {
        cout << i << " ";
    }
    cout << endl;
    vector<double> costs = {0.1, 0.2, 0.0, -1.};
    vector<double> lambda_c(costs.size());
    std::transform(costs.begin(), costs.end(), lambda_c.begin(),
                   std::bind1st(std::multiplies<double>(), 0.5));
    for (double &i : lambda_c) {
        std::cout << i << ' ';
    }
    for (double &i : costs) {
        std::cout << i << ' ';
    }

}


// create a grid graph.
pair<vector<pair<int, int> >, vector<double>>
get_grid_graph(int num_nodes, bool is_rand_wei) {
    // make them empty.
    vector<pair<int, int> > edges;
    vector<double> weights;
    int index = 0;
    auto length = (int) sqrt(num_nodes);
    // generate a graph which is a grid graph.
    int width = length;
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < width; j++) {
            pair<int, int> edge;
            if ((index % length) != (length - 1)) {
                edge = make_pair(index, index + 1);
                edges.push_back(edge);
                if ((index + length) < num_nodes) {
                    edge = make_pair(index, index + length);
                    edges.push_back(edge);
                }
            } else {
                if ((index + length) < num_nodes) {
                    edge = make_pair(index, index + length);
                    edges.push_back(edge);
                }
            }
            index += 1;
        }
    }
    for (size_t i = 0; i < edges.size(); i++) {
        weights.push_back(1.0);
    }
    return make_pair(edges, weights);
}

// Test 1,000,000 nodes. To test the scalability of fast-pcst.
void test_fast_pcst() {
    int num_nodes = 10000;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.01, 0.1);
    vector<double> prizes;
    for (int i = 0; i < num_nodes; i++) {
        prizes.push_back(dist(mt));
    }
    std::uniform_int_distribution<int> dist2(10, 100);
    for (int j = 0; j < 1000; j++) {
        prizes[dist2(mt)] = 10.0;
    }
    cout << endl << "starting ... run pcsf " << endl;
    const clock_t start_time = clock();
    vector<pair<int, int> > edges;
    vector<double> costs;
    pair<vector<pair<int, int> >, vector<double>>
            graph = get_grid_graph(num_nodes, false);
    edges = graph.first;
    costs = graph.second;
    PCSTFast algo(edges, prizes, costs, PCSTFast::kNoRoot, 1,
                  PCSTFast::kGWPruning, 0, nullptr);
    vector<int> node_result;
    vector<int> edge_result;
    algo.run(&node_result, &edge_result);
    double ms_unit = (CLOCKS_PER_SEC / 1000.);
    double run_time = (float(clock() - start_time) / ms_unit);
    cout << "number of nodes: " << node_result.size() << endl;
    cout << "number of edges: " << edge_result.size() << endl;
    cout << "run time: " << run_time << "ms" << endl;
    cout << "finish test..." << endl;
}


void test_tail_proj() {
    pair<vector<pair<int, int> >, vector<double>> graph;
    graph = get_grid_graph(100, false);
    vector<pair<int, int> > edges = graph.first;
    vector<double> costs = graph.second;
}

vector<int> get_nodes_map(vector<int> nodes) {
    int max = *max_element(nodes.begin(), nodes.end());
    // nodes_map to find the index of one node.
    vector<int> nodes_map((size_t) (max + 1));
    for (int index = 0; index < nodes.size(); index++) {
        nodes_map[nodes[index]] = index;
    }
    return nodes_map;
}

pair<vector<int>, vector<int>> dfs_tour(vector<int> nodes_t,
                                        vector<int> edges_t,
                                        vector<pair<int, int>> edges) {
    vector<vector<tuple<int, int, bool>>> adj; // adj: <original,neighbors>
    for (auto const &node: nodes_t) {
        vector<tuple<int, int, bool>>
                neighbors;
        adj.push_back(neighbors);
    }
    vector<int> nodes_map = get_nodes_map(nodes_t); // node to index map
    for (auto const &edge_index:edges_t) {
        int index_u = nodes_map[edges[edge_index].first];
        int index_v = nodes_map[edges[edge_index].second];
        // each tuple is: (indexed node, edge_index, is_visited)
        tuple<int, int, bool> nei_v = make_tuple(index_v, edge_index, false);
        tuple<int, int, bool> nei_u = make_tuple(index_u, edge_index, false);
        adj[index_u].push_back(nei_v); // edge u --> v
        adj[index_v].push_back(nei_u); // edge v --> u
    }

    // ------------------ Euler Tour algorithm --------------------
    vector<int> tour_nodes;
    vector<int> tour_edges;
    int start_node = nodes_map[nodes_t[0]]; // The first element as root.
    vector<bool> visited(adj.size(), false);
    tour_nodes.push_back(start_node);
    while (true) {
        bool flag_1 = false;
        visited[start_node] = true;
        // iterate the neighbors of each node. in this loop, we check if
        // there exists any its neighbor which has not been visited.
        for (int i = 0; i < adj[start_node].size(); i++) {
            int next_node = get<0>(adj[start_node][i]);
            int edge_index = get<1>(adj[start_node][i]);
            if (!visited[next_node]) { // first time to visit this node.
                visited[next_node] = true; // mark it as labeled.
                tour_nodes.push_back(next_node);
                tour_edges.push_back(edge_index);
                get<2>(adj[start_node][i]) = true; // mark it as labeled.
                start_node = next_node;
                flag_1 = true;
                break;
            }
        }
        // all neighbors are visited. Then we check if
        // there exists neighbors which is false nodes.
        if (!flag_1) {
            bool flag_2 = false;
            for (size_t i = 0; i < adj[start_node].size(); i++) {
                int next_node = get<0>(adj[start_node][i]);
                int edge_index = get<1>(adj[start_node][i]);
                bool is_visited = get<2>(adj[start_node][i]);
                if (!is_visited) { // there exists a neighbor. has false node
                    get<2>(adj[start_node][i]) = true;
                    tour_nodes.push_back(next_node);
                    tour_edges.push_back(edge_index);
                    start_node = next_node;
                    flag_2 = true;
                    break;
                }
            }
            if (!flag_2) {
                break;// all nodes are visited and there is no false nodes.
            }
        }
    }
    // convert indexed nodes back to original nodes.
    for (int &tour_node : tour_nodes) {
        int original_node = nodes_t[tour_node];
        tour_node = original_node;
    }
    return make_pair(tour_nodes, tour_edges);
}

// greater struct is for sort in descending order.
struct greater {
    template<class T>
    bool operator()(T const &a, T const &b) const { return a > b; }
};


vector<pair<vector<int>, vector<int>>>
sort_forest(pair<vector<int>, vector<int>> f,
            vector<pair<int, int>> edges,
            vector<double> prizes,
            vector<double> costs) {
    // Create a descending ordered forest.
    vector<int> nodes_f = f.first;
    vector<int> edges_f = f.second;
    vector<int> nodes_map = get_nodes_map(nodes_f);
    // Create a adj graph. Node ID starts from 0.
    vector<vector<int>> adj;
    for (size_t index = 0; index < nodes_f.size(); index++) {
        adj[index] = vector<int>();
    }
    for (auto const &edge_index:f.second) {
        int index_u = nodes_map[edges[edge_index].first];
        int index_v = nodes_map[edges[edge_index].second];
        adj[index_u].emplace_back(index_v);
        adj[index_v].emplace_back(index_u);
    }
    int t = 0; // component id
    vector<int> comp(adj.size(), 0); //label the nodes to the components id.
    vector<bool> visited(adj.size(), false);
    for (auto i = 0; i < adj.size(); i++) { // dfs algorithm to get cc
        if (!visited[i]) {
            t++; // to label component id.
            vector<int> stack(0);
            stack.push_back(i);
            while (!stack.empty()) {
                int s = stack.back();
                stack.pop_back();
                if (!visited[s]) {
                    visited[s] = true;
                    comp[s] = t;
                }
                for (auto k: adj[s]) {
                    if (!visited[k]) {
                        stack.push_back(k);
                    }
                }
            }
        }
    }
    int num_cc = t; //the total number of connected components in the forest.
    vector<pair<vector<int>, vector<int>>> trees; // A collection of trees.
    vector<double> trees_pi((size_t) num_cc, 0.0); // prize of each tree
    vector<double> trees_cost((size_t) num_cc, 0.0); // cost of each tree
    for (int i = 0; i < num_cc; i++) { // initialize trees.
        vector<int> nodes_;
        vector<int> edges_;
        pair<vector<int>, vector<int>> tree = make_pair(nodes_, edges_);
        trees.push_back(tree);
    }
    for (size_t node = 0; node < nodes_f.size(); node++) {
        int tree_index = comp[node]; // construct each tree
        // add a node into the corresponding tree.
        int ori_node = nodes_f[node]; //convert it to original node.
        trees[tree_index].first.push_back(ori_node);
        trees_pi[tree_index] += prizes[ori_node];
    }
    for (auto const &edge_index:edges_f) { // try to insert edges into trees.
        int u = edges[edge_index].first; // random select one endpoint
        int index_u = nodes_map[u];
        trees[comp[index_u]].second.push_back(edge_index);
        trees_cost[comp[index_u]] += costs[edge_index];
    }
    vector<pair<double, int> > trees_wei; // weights for sorting.
    for (int i = 0; i < num_cc; i++) {
        double weight;
        if (trees_cost[i] > 0.0) {
            weight = trees_pi[i] / trees_cost[i];
        } else {
            weight = numeric_limits<double>::max(); // for a single node tree
        }
        pair<double, int> wei = make_pair(weight, i);
        trees_wei.push_back(wei);
    }
    sort(trees_wei.begin(), trees_wei.end(), greater());
    vector<pair<vector<int>, vector<int>>> sorted_trees;
    for (auto const &sorted_wei:trees_wei) {
        int i = sorted_wei.second;
        pair<vector<int>, vector<int>> tree_i = trees[i];
        sorted_trees.emplace_back(tree_i);
    }
    return sorted_trees;
}

void test_dfs_tour() {
    // -------------- test case 1 -------------------------
    vector<pair<int, int> > edges = vector<pair<int, int> >();
    vector<int> nodes_t = {1, 2, 3, 4, 5, 6};
    vector<int> edges_t = {0, 1, 2, 3, 4};
    edges.emplace_back(1, 2);
    edges.emplace_back(1, 3);
    edges.emplace_back(1, 4);
    edges.emplace_back(3, 5);
    edges.emplace_back(3, 6);
    pair<vector<int>, vector<int>> tour;
    tour = dfs_tour(nodes_t, edges_t, edges);
    cout << "tour nodes: ";
    for (auto node : tour.first) {
        cout << node << " ";
    }
    cout << endl << "tour edges: ";
    for (auto edge : tour.second) {
        cout << edge << " ";
    }
    cout << endl;
    // -------------- test case 2 -------------------------
    edges = vector<pair<int, int> >();
    nodes_t = {1, 2, 3, 4, 5};
    edges_t = {0, 1, 2, 3};
    edges.emplace_back(1, 2);
    edges.emplace_back(1, 3);
    edges.emplace_back(1, 4);
    edges.emplace_back(1, 5);
    tour = dfs_tour(nodes_t, edges_t, edges);
    cout << "tour nodes: ";
    for (auto node : tour.first) {
        cout << node << " ";
    }
    cout << endl << "tour edges: ";
    for (auto edge : tour.second) {
        cout << edge << " ";
    }
    cout << endl;

    // -------------- test case 3 -------------------------
    edges = vector<pair<int, int> >();
    nodes_t = {6, 7, 8, 9, 10};
    edges_t = {0, 1, 2, 3};
    edges.emplace_back(6, 7);
    edges.emplace_back(7, 8);
    edges.emplace_back(8, 9);
    edges.emplace_back(9, 10);
    tour = dfs_tour(nodes_t, edges_t, edges);
    cout << "tour nodes: ";
    for (auto node : tour.first) {
        cout << node << " ";
    }
    cout << endl << "tour edges: ";
    for (auto edge : tour.second) {
        cout << edge << " ";
    }
    cout << endl;

    // -------------- test case 4 -------------------------
    edges = vector<pair<int, int> >();
    nodes_t = {3, 4, 5, 6, 7, 9, 8, 15, 16, 12, 11, 10, 13, 14};
    edges_t = {8, 7, 6, 5, 4, 1, 2, 3, 0, 11, 13, 15, 17};

    edges.emplace_back(11, 10);
    edges.emplace_back(11, 12);
    edges.emplace_back(12, 13);
    edges.emplace_back(12, 14);
    edges.emplace_back(5, 11);
    edges.emplace_back(5, 7);
    edges.emplace_back(4, 6);
    edges.emplace_back(4, 5);
    edges.emplace_back(3, 4);
    edges.emplace_back(-1, -1); // fake edge
    edges.emplace_back(-1, -1); // fake edge
    edges.emplace_back(15, 16);
    edges.emplace_back(-1, -1); // fake edge
    edges.emplace_back(8, 15);
    edges.emplace_back(-1, -1); // fake edge
    edges.emplace_back(7, 8);
    edges.emplace_back(-1, -1); // fake edge
    edges.emplace_back(7, 9);

    tour = dfs_tour(nodes_t, edges_t, edges);
    cout << "tour nodes: ";
    for (auto node : tour.first) {
        cout << node << " ";
    }
    cout << endl << "tour edges: ";
    for (auto edge : tour.second) {
        cout << edge << " ";
    }
    cout << endl;
}

void test_sort_forest() {
    cout << " test ";
    int num_nodes = 16;
    vector<double> prizes;
    for (int i = 0; i < num_nodes; i++) {
        prizes.push_back(1.0);
    }
    vector<pair<int, int> > edges;
    vector<double> costs;
    pair<vector<pair<int, int> >, vector<double>> graph;
    graph = get_grid_graph(num_nodes, false);
}

int main() {
    test_sort_forest();
    test_dfs_tour();
    main_test();
    test_fast_pcst();
}