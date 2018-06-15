/*============================================================================
 * This head-oracle is an algorithm proposed in the paper titled:
 * "A Nearly-Linear Time Framework for Graph-Structured Sparsity".
 * Created by Baojian Zhou, Email: bzhou6@albany.edu
 * Date: 05/25/2017
============================================================================*/
#include <limits>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "pcst_fast.h"
#include "head_tail.h"


using std::min;
using std::cout;
using std::sort;
using std::pair;
using std::endl;
using std::get;
using std::tuple;
using std::vector;
using std::make_pair;
using std::make_tuple;
using std::numeric_limits;
using cluster_approx::PCSTFast;

HeadApprox::HeadApprox(const vector<pair<int, int>> &edges_,
                       const vector<double> &costs_,
                       const vector<double> &prizes_,
                       int g_, int s_, double C_, double delta_)
        : edges(edges_), costs(costs_), prizes(prizes_),
          g(g_), s(s_), C(C_), delta(delta_), verbose_level(0) {
}


pair<vector<int>, vector<int>> HeadApprox::run() {
    double pi_min = min_pi(); // total_prizes will be calculated.
    double lambda_r = (2. * C) / (pi_min);
    double lambda_l = 1. / (4. * total_prizes);
    double epsilon = (delta * C) / (2. * total_prizes);
    if (verbose_level > 0) {
        cout << "lambda_r: " << lambda_r << endl;
        cout << "lambda_l: " << lambda_l << endl;
        cout << "total_prizes: " << total_prizes << endl;
        cout << "epsilon: " << epsilon << endl;
    }
    Forest f;
    f = pcsf_gw(pi_lambda(lambda_r));
    if (cost_f(f) <= (2. * C)) {
        return f;
    }// ensure that we have invariant c(F_r) > 2 C
    while ((lambda_r - lambda_l) > epsilon) {
        double lambda_m = (lambda_l + lambda_r) / 2.;
        if (verbose_level > 0) {
            cout << "lambda_r: " << lambda_r << ", lambda_l: " << lambda_l;
            cout << ", lambda_m: " << lambda_m << ", gap: ";
            cout << (lambda_r - lambda_l) << endl;
        }
        f = pcsf_gw(pi_lambda(lambda_m));
        if (cost_f(f) > (2. * C)) {
            lambda_r = lambda_m;
        } else {
            lambda_l = lambda_m;
        }
    } // binary search over the Lagrange parameter lambda
    Forest f_l = pcsf_gw(pi_lambda(lambda_l));
    Forest f_r = pcsf_gw(pi_lambda(lambda_r));
    // It seems that this method is not necessary.
    Forest f_r_prime = prune_forest(f_r);
    if (pi_f(f_l) >= pi_f(f_r_prime)) {
        return f_l;
    } else {
        return f_r;
    }
}

double HeadApprox::min_pi() {
    total_prizes = 0.0;
    double positive_min = std::numeric_limits<double>::max();
    for (auto const &val: prizes) {
        total_prizes += val;
        if ((val < positive_min) && (val > 0.0)) {
            positive_min = val;
        }
    }
    /**
     *Warning: There is a precision issue here. We may need to define a
     * minimum precision. Since, in our experiment,  we found that some very
     * small positive number could be like 1.54046e-310, 1.54046e-310.
     * In this case, the fast-pcst cannot stop!!!
     */
    if (positive_min < 1e-8) {
        if (verbose_level > 0) {
            cout << "warning too small positive value found " << endl;
        }
        positive_min = 1e-8;
    }
    return positive_min;
}

vector<double> HeadApprox::pi_lambda(double lambda) { // pi * lambda
    vector<double> pi_lambda(prizes.size());
    std::transform(prizes.begin(), prizes.end(), pi_lambda.begin(),
                   std::bind1st(std::multiplies<double>(), lambda));
    return pi_lambda;
}

pair<vector<int>, vector<int>> HeadApprox::prune_forest(Forest f) {
    // case 1: usually, there is only one tree.
    if (g == 1) {
        if (cost_f(f) <= C) {
            return f;
        } else if (0.0 < C) {
            return prune_tree(f, C);
        } else {
            return argmax_tree(f);
        }
    }
    // case 2: there are at least two trees.
    vector<HeadApprox::Tree> sorted_f = sort_forest(f);
    f.first.clear(); //clear nodes_f and edges_f, and then update them.
    f.second.clear();
    double c_r = C;
    for (const auto &tree_i : sorted_f) {
        double c_tree_i = cost_f(tree_i);
        Tree tree_i_prime;
        if (c_r >= c_tree_i) {
            tree_i_prime = tree_i;
            c_r = c_r - c_tree_i;
        } else if (c_r > 0.0) {
            // tree_i must have at least two nodes and one edge.
            tree_i_prime = prune_tree(tree_i, c_r);
            c_r = 0.0;
        } else {
            tree_i_prime = argmax_tree(tree_i);// A single node tree.
        }
        for (auto const &node:tree_i_prime.first) { // add tree_i_prime.
            f.first.push_back(node);
        }
        for (auto const &edge_index:tree_i_prime.second) {
            f.second.push_back(edge_index);
        }
    }// iterate trees by descending order.
    return f;
}

vector<HeadApprox::Tree> HeadApprox::sort_forest(Forest f) {
    // Create a descending ordered forest.
    vector<int> nodes_map = get_nodes_map(f.first);
    // Create a adj graph. Node ID starts from 0.
    vector<vector<int>> adj;
    for (size_t index = 0; index < f.first.size(); index++) {
        vector<int> nei_list;
        adj.push_back(nei_list);
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
            vector<int> stack;
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
            t++; // to label component id.
        }
    }
    int num_cc = t; //the total number of connected components in the forest.
    vector<HeadApprox::Tree> trees; // A collection of trees.
    vector<double> trees_pi((size_t) num_cc, 0.0); // prize of each tree
    vector<double> trees_cost((size_t) num_cc, 0.0); // cost of each tree
    for (int i = 0; i < num_cc; i++) { // initialize trees.
        vector<int> nodes_;
        vector<int> edges_;
        Tree tree = make_pair(nodes_, edges_);
        trees.push_back(tree);
    }
    for (size_t node = 0; node < f.first.size(); node++) {
        int tree_index = comp[node]; // construct each tree
        // add a node into the corresponding tree.
        int ori_node = f.first[node]; //convert it to original node.
        trees[tree_index].first.push_back(ori_node);
        trees_pi[tree_index] += prizes[ori_node];
    }
    for (auto const &edge_index:f.second) { // try to insert edges into trees.
        int u = edges[edge_index].first; // random select one endpoint
        int index_u = nodes_map[u];
        trees[comp[index_u]].second.push_back(edge_index);
        trees_cost[comp[index_u]] += costs[edge_index];
    }
    vector<pair<double, int>> trees_wei; // weights for sorting.
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
    vector<HeadApprox::Tree> sorted_trees;
    for (auto const &sorted_wei:trees_wei) {
        int i = sorted_wei.second;
        Tree tree_i = trees[i];
        sorted_trees.emplace_back(tree_i);
    }
    return sorted_trees;
}

// nodes_t has at least two nodes.
HeadApprox::Tree HeadApprox::prune_tree(Tree t, double c_prime) {
    vector<int> nodes_t = t.first;
    vector<int> edges_t = t.second;
    pair<vector<int>, vector<int>> tour = dfs_tour(nodes_t, edges_t);
    vector<double> pi_prime; // calculating pi_prime.
    int maximum_node = *max_element(nodes_t.begin(), nodes_t.end());
    vector<bool> tmp_vector((size_t) (maximum_node + 1), true);
    for (int v_j : tour.first) {
        if (tmp_vector[v_j]) { // first time show in the tour.
            pi_prime.push_back(prizes[v_j]);
            tmp_vector[v_j] = false;
        } else {
            pi_prime.push_back(0.0);
        }
    }
    nodes_t.clear();
    edges_t.clear();
    double phi = pi_f(t) / cost_f(t);
    for (auto const &node:tour.first) {
        if (prizes[node] >= ((c_prime * phi) / 6.)) {
            nodes_t.push_back(node); // create a single node tree.
            return make_pair(nodes_t, edges_t);
        }
    }
    vector<size_t> p_l;
    for (size_t i = 0; i < tour.first.size(); i++) {
        p_l.push_back(i);
        double pi_prime_pl = 0.0;
        for (auto const &ind:p_l) {
            pi_prime_pl += pi_prime[ind];
        }
        double c_prime_pl = 0.0;
        if (p_l.size() >= 2) { // <= 1: there is no edge.
            for (size_t j = 0; j < p_l.size() - 1; j++) {
                c_prime_pl += costs[tour.second[p_l[j]]];
            }
        }
        if (c_prime_pl > c_prime) { // start a new sublist
            p_l.clear();
        } else if (pi_prime_pl >= ((c_prime * phi) / 6.)) {
            for (auto const &ind:p_l) {
                nodes_t.push_back(tour.first[ind]);
                edges_t.push_back(tour.second[ind]);
            }
            return make_pair(nodes_t, edges_t);
        }
    }
    cout << "Never reach at this point." << endl; //Merge procedure.
    exit(0);
}

// Lagrange's parameter is on prizes.
pair<vector<int>, vector<int>> HeadApprox::pcsf_gw(vector<double> prizes) {
    PCSTFast algo(edges, prizes, costs, PCSTFast::kNoRoot, g,
                  PCSTFast::kGWPruning, 0, nullptr);
    if (!algo.run(&result_nodes, &result_edges)) {
        cout << "Error: run pcst_fast error." << endl;
        exit(0);
    }
    return make_pair(result_nodes, result_edges);
}

double HeadApprox::pi_f(Forest f) {
    double pi_f = 0.0;
    for (auto &node:f.first) {
        pi_f += prizes[node];
    }
    return pi_f;
}

double HeadApprox::cost_f(Forest f) {
    double cost_f = 0.0;
    for (auto &edge_index:f.second) {
        cost_f += costs[edge_index];
    }
    return cost_f;
}

/**
 * This method is to find a euler tour given a tree. This method is
 * proposed in the following paper:
 * Authors : Edmonds, Jack, and Ellis L. Johnson.
 * Title : Matching, Euler tours and the Chinese postman
 * Journal: Mathematical programming 5.1 (1973): 88-124.
 */
pair<vector<int>, vector<int>> HeadApprox::dfs_tour(
        const vector<int> nodes_t, const vector<int> edges_t) {
    //Make sure the tree has at least two nodes.
    if (nodes_t.size() <= 1) {
        cout << "error: The input tree has at least two nodes.";
        exit(0);
    }
    vector<vector<tuple<int, int, bool >>>
            adj; // adj: <original,neighbors>
    for (auto const &node: nodes_t) {
        vector<tuple<int, int, bool>> neighbors;
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
    return make_pair(tour_nodes, tour_edges); //original nodes, edge indices
}

HeadApprox::Tree HeadApprox::argmax_tree(HeadApprox::Tree tree) {
    // create a single node tree.
    int maximum_node = tree.first[0];
    double maximum_pi = prizes[maximum_node];
    for (auto const &node:tree.first) {
        if (maximum_pi < prizes[node]) {
            maximum_pi = prizes[node];
            maximum_node = node;
        }
    }
    vector<int> single_tree = {maximum_node};
    vector<int> empty_edges;
    return make_pair(single_tree, empty_edges);
}

vector<int> HeadApprox::get_nodes_map(vector<int> nodes) {
    int max = *max_element(nodes.begin(), nodes.end());
    // nodes_map to find the index of one node.
    vector<int> nodes_map((size_t) (max + 1));
    for (int index = 0; index < nodes.size(); index++) {
        nodes_map[nodes[index]] = index;
    }
    return nodes_map;
}


HeadApprox::~HeadApprox() = default;


TailApprox::TailApprox(const vector<pair<int, int>> &edges_,
                       const vector<double> &costs_,
                       const vector<double> &prizes_,
                       const int g_,
                       const int s_,
                       const double C_,
                       const double nu_,
                       const double delta_) :
        edges(edges_), costs(costs_), prizes(prizes_),
        g(g_), s(s_), C(C_), nu(nu_), delta(delta_), verbose_level(0) {
}


pair<vector<int>, vector<int>> TailApprox::run() {
    //prizes pi[i] = b_i * b_i, all other parameters
    //are settled in the initialization.
    double pi_min = min_pi();
    double lambda_0 = pi_min / (2.0 * C);
    pcsf_gw(cost_lambda(lambda_0));
    double c_f = cost_F(result_edges);
    double pi_f_bar = pi_F_bar(result_nodes);
    if ((c_f <= (2.0 * C)) && (pi_f_bar <= 0.0)) {
        return make_pair(result_nodes, result_edges);
    }
    double lambda_r = 0.;
    double lambda_l = 3. * total_prizes;
    double epsilon = (pi_min * delta) / C;
    double lambda_m;
    vector<double> c_l;
    while ((lambda_l - lambda_r) > epsilon) {
        lambda_m = (lambda_l + lambda_r) / 2.;
        pcsf_gw(cost_lambda(lambda_m));
        c_f = cost_F(result_edges);
        if ((c_f >= (2. * C)) && (c_f <= (nu * C))) {
            return make_pair(result_nodes, result_edges);
        }
        if (c_f >= (nu * C)) {
            lambda_r = lambda_m;
        } else {
            lambda_l = lambda_m;
        }
    } // while
    pcsf_gw(cost_lambda(lambda_l));
    return make_pair(result_nodes, result_edges);
}


double TailApprox::min_pi() {
    // return minimum positive value of prizes of nodes.
    total_prizes = 0.0;
    double positive_min = std::numeric_limits<double>::max();
    for (auto const &val: prizes) {
        total_prizes += val;
        if ((val < positive_min) && (val > 0.0)) {
            positive_min = val;
        }
    }
    /**
     *Warning: There is a precision issue here. We may need to define a
     * minimum precision. Since, in our experiment,  we found that some very
     * small positive number could be like 1.54046e-310, 1.54046e-310.
     * In this case, the fast-pcst cannot stop!!!
     */
    if (positive_min < 1e-8) {
        if (verbose_level > 0) {
            cout << "warning too small positive value found " << endl;
        }
        positive_min = 1e-8;
    }
    return positive_min;
}

// Run fast-pcst method.
bool TailApprox::pcsf_gw(const vector<double> &costs) {
    PCSTFast algo(edges, prizes, costs, PCSTFast::kNoRoot,
                  g, PCSTFast::kGWPruning, 0, nullptr);
    return algo.run(&result_nodes, &result_edges);
}

// calculate c(e)*lambda for each edge.
vector<double> TailApprox::cost_lambda(double lambda_) {
    vector<double> cost_lambda(costs.size());
    std::transform(costs.begin(), costs.end(), cost_lambda.begin(),
                   std::bind1st(std::multiplies<double>(), lambda_));
    return cost_lambda;
}

// calculate c(F)
double TailApprox::cost_F(std::vector<int> result_edges) {
    double cost_f = 0.0;
    for (auto const &val:result_edges) {
        cost_f += costs[val];
    }
    return cost_f;
}

// calculate pi(\bar{F})
double TailApprox::pi_F_bar(std::vector<int> result_nodes) {
    double pi_f = 0.0;
    for (auto const &node:result_nodes) {
        pi_f += prizes[node];
    }
    return (total_prizes - pi_f);
}


/*****************************************************************************
 *
 * This function is a Python interface.
 * @param self just a NULL object.
 * @param args
 *          edges: edges.shape=(m,2), where m is # of edges in the graph.
 *          edge_weights: edges.shape=(m, ) edge costs.
 *          g: int
 *          s: int
 *          B: double
 *          delta: double
 *          root:                  int, root for pcst, -1 (no root).
 *          num_active_cluster:    int, number of forest, default is 1.
 *          pruning                string, ['strong','gw']. default is 'strong'
 *          verbose_level:         int, >=2 print more details.
 * @return
 ****************************************************************************/
static PyObject *head_proj(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error.");
        exit(0);
    }
    PyArrayObject *input_edges, *input_edge_costs;
    unsigned int s_low, s_high, edge_costs_multiplier, root;
    unsigned int num_active_cluster, verbose_level, max_num_iter;
    char *pathway_id, *pruning;
    if (!PyArg_ParseTuple(args, "O!O!iiizii",
                          &PyArray_Type, &input_edges,
                          &PyArray_Type, &input_edge_cost,
                          &edge_costs_multiplier, &root,
                          &num_active_cluster, &pruning,
                          &verbose_level)) {
        printf("something wrong");
        exit(-1);
    }
    // build some parameters
    long int len_edges = PyArray_Size((PyObject *) input_edges);
    long int len_edge_costs = PyArray_Size((PyObject *) input_edge_costs);
    //1. edges are the same.
    vector<pair<int, int>> edges;
    for (int i = 0; i < len(edges_0); i++) {
        int edge_0 = boost::python::extract<int>(edges_0[i]);
        int edge_1 = boost::python::extract<int>(edges_1[i]);
        edges.emplace_back(edge_0, edge_1);
    }
    //2. prizes.
    vector<double> prizes;
    for (int i = 0; i < len(b_); i++) {
        double b_i = boost::python::extract<double>(b_[i]);
        prizes.push_back(b_i * b_i);
    }
    //3. cost c(e) = w(e) + B / s
    vector<double> costs;
    for (int i = 0; i < len(weights_); i++) {
        double w = boost::python::extract<double>(weights_[i]);
        costs.push_back(w + B / (s * 1.0));
    }
    //4. cost budget C
    double C = 2. * B;
    HeadApprox head(edges, costs, prizes, g, s, C, delta);
    pair<vector<int>, vector<int>> f = head.run();
    //5. package the result nodes and edges.
    boost::python::list result;
    for (auto const &node:f.first) {
        result.append(node);
    }
    result.append(-1);
    for (auto const &edge_index:f.second) {
        result.append(edge_index);
    }
    return result;
    PyObject *result = run_algo(input_data);
    if (result == NULL) {
        perror("something wrong !");
        exit(0);
    }
    // to free memory
    free(input_data);
    free(edge_costs);
    free(edges);
    free(w0);
    free(y);
    free(x);
    return result;
}


/*****************************************************************************
 *
 * // This tail approximation method is a wrapper for Python interface.
// Notice that you need to check the following parts:
// 1. Make sure the graph is connected.
// 2. Make sure the input vector is not a zero vector.
 * This function is a Python interface.
 * @param self just a NULL object.
 * @param args
 *          edges: edges.shape=(m,2), where m is # of edges in the graph.
 *          edge_weights: edges.shape=(m, ) edge costs.
 *          g: int
 *          s: int
 *          B: double
 *          delta: double
 *          root:                  int, root for pcst, -1 (no root).
 *          num_active_cluster:    int, number of forest, default is 1.
 *          pruning                string, ['strong','gw']. default is 'strong'
 *          verbose_level:         int, >=2 print more details.
 * @return
 ****************************************************************************/
static PyObject *tail_proj(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error.");
        exit(0);
    }
    PyArrayObject *input_edges, *input_edge_costs;
    unsigned int s_low, s_high, edge_costs_multiplier, root;
    unsigned int num_active_cluster, verbose_level, max_num_iter;
    char *pathway_id, *pruning;
    if (!PyArg_ParseTuple(args, "O!O!iiizii",
                          &PyArray_Type, &input_edges,
                          &PyArray_Type, &input_edge_cost,
                          &edge_costs_multiplier, &root,
                          &num_active_cluster, &pruning,
                          &verbose_level)) {
        printf("something wrong");
        exit(-1);
    }
    //1. edges are the same.
    vector<pair<int, int>> edges;
    for (int i = 0; i < len(edges_0); i++) {
        int edge_0 = boost::python::extract<int>(edges_0[i]);
        int edge_1 = boost::python::extract<int>(edges_1[i]);
        edges.emplace_back(edge_0, edge_1);
    }
    //2. prizes.
    vector<double> prizes;
    for (int i = 0; i < len(b_); i++) {
        double b_i = boost::python::extract<double>(b_[i]);
        prizes.push_back(b_i * b_i);
    }
    //3. cost c(e) = w(e) + B / s.
    vector<double> costs;
    for (int i = 0; i < len(weights_); i++) {
        double w = boost::python::extract<double>(weights_[i]);
        costs.push_back(w + B / (s * 1.0));
    }
    //4. cost budget C.
    double C = 2. * B;
    //5. delta = min(0.5,1/nu)
    double delta = min(0.5, 1. / nu);
    TailApprox tail(edges, costs, prizes, g, s, C, nu, delta);
    pair<vector<int>, vector<int>> f = tail.run();
    boost::python::list result;
    for (auto const &node:f.first) {
        result.append(node);
    }
    result.append(-1);
    for (auto const &edge_index:f.second) {
        result.append(edge_index);
    }
    return result;
}


/*****************************************************************************
 *
 * This function is a Python interface.
 * @param self just a NULL object.
 * @param args
 *          edges: edges.shape=(m,2), where m is # of edges in the graph.
 *          edge_weights: edges.shape=(m, ) edge costs.
 *          g: int
 *          s: int
 *          B: double
 *          delta: double
 *          root:                  int, root for pcst, -1 (no root).
 *          num_active_cluster:    int, number of forest, default is 1.
 *          pruning                string, ['strong','gw']. default is 'strong'
 *          verbose_level:         int, >=2 print more details.
 * @return
 ****************************************************************************/
static PyObject *fast_pcst(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error.");
        exit(0);
    }
    PyArrayObject *input_edges, *input_edge_costs;
    unsigned int s_low, s_high, edge_costs_multiplier, root;
    unsigned int num_active_cluster, verbose_level, max_num_iter;
    char *pathway_id, *pruning;
    if (!PyArg_ParseTuple(args, "O!O!iiizii",
                          &PyArray_Type, &input_edges,
                          &PyArray_Type, &input_edge_cost,
                          &edge_costs_multiplier, &root,
                          &num_active_cluster, &pruning,
                          &verbose_level)) {
        printf("something wrong");
        exit(-1);
    }
    // build some parameters
    long int len_edges = PyArray_Size((PyObject *) input_edges);
    long int len_edge_costs = PyArray_Size((PyObject *) input_edge_costs);
    //1. edges of the graph.
    vector<pair<int, int>> edges;
    for (int i = 0; i < len(edges_0); i++) {
        int edge_0 = boost::python::extract<int>(edges_0[i]);
        int edge_1 = boost::python::extract<int>(edges_1[i]);
        edges.emplace_back(edge_0, edge_1);
    }
    //2. costs of the edges
    vector<double> costs;
    for (int i = 0; i < len(costs_); i++) {
        double cost = boost::python::extract<double>(costs_[i]);
        costs.push_back(cost);
    }
    //3. prizes of the graph.
    vector<double> prizes;
    for (int i = 0; i < len(prizes_); i++) {
        double prize = boost::python::extract<double>(prizes_[i]);
        prizes.push_back(prize);
    }
    PCSTFast pcst_pcst(edges, prizes, costs, PCSTFast::kNoRoot, g,
                       PCSTFast::kGWPruning, 0, nullptr);
    vector<int> result_nodes;
    vector<int> result_edges;
    pcst_pcst.run(&result_nodes, &result_edges);
    boost::python::list result;
    for (auto const &node:result_nodes) {
        result.append(node);
    }
    result.append(-1);
    for (auto const &edge_index:result_edges) {
        result.append(edge_index);
    }
    return result;
}


/*  define functions in module */
static PyMethodDef HeadProjMethods[] = {
        {"head_proj", head_proj, METH_VARARGS, "head projection docs"},
        {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,
        "head_proj", "Some documentation",
        -1,
        HeadProjMethods
};

PyMODINIT_FUNC PyInit_cos_module(void) {
    return PyModule_Create(&cModPyDem);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC inithead_proj(void) {
    import_array();
    (void) Py_InitModule("head_proj", HeadProjMethods);
}

#endif