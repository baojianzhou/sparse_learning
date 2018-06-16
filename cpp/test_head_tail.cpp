//
// Created by baojian on 6/16/18.
//
#include <iostream>
#include "head_tail.h"

using std::make_pair;

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

void test_tail_proj() {
    int num_nodes = 100, g = 1, s = 10;
    double budget = 10., delta = 1. / 169.;
    pair<vector<pair<int, int> >, vector<double>> graph;
    graph = get_grid_graph(num_nodes, false);
    //1. edges are the same.
    vector<pair<int, int> > edges = graph.first;
    vector<double> costs = graph.second;
    cout << "test for tail projection " << endl;
    //2. prizes.
    vector<double> prizes;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.1, 5.0);
    for (int i = 0; i < num_nodes; i++) {
        prizes.push_back(dist(mt));
    }
    std::uniform_int_distribution<int> dist2(20, 50);
    for (int j = 0; j < 10; j++) {
        prizes[dist2(mt)] = 10.0;
    }
    //4. cost budget C
    double C = 2. * budget;
    HeadApprox head(edges, costs, prizes, g, s, C, delta);
    pair<vector<int>, vector<int>> f = head.run();
    //5. package the result nodes and edges.
    cout << "node result: " << endl;
    for (auto const &node:f.first) {
        cout << node << " ";
    }
    cout << endl;
    cout << "edge result: " << endl;
    for (auto const &edge_index:f.second) {
        cout << edge_index << " ";
    }
    cout << endl;
}

void test_head_proj() {
    int num_nodes = 100, g = 1, s = 10;
    double budget = 10., nu = 2.5;
    pair<vector<pair<int, int> >, vector<double>> graph;
    graph = get_grid_graph(num_nodes, false);
    //1. edges are the same.
    vector<pair<int, int> > edges = graph.first;
    vector<double> costs = graph.second;
    cout << "test for head projection " << endl;
    //2. prizes.
    vector<double> prizes;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.1, 5.0);
    for (int i = 0; i < num_nodes; i++) {
        prizes.push_back(dist(mt));
    }
    std::uniform_int_distribution<int> dist2(20, 50);
    for (int j = 0; j < 10; j++) {
        prizes[dist2(mt)] = 10.0;
    }
    //4. cost budget C
    double C = 2. * budget;
    double delta = min(0.5, 1. / nu);
    TailApprox tail(edges, costs, prizes, g, s, C, nu, delta);
    pair<vector<int>, vector<int>> f = tail.run();
    //5. package the result nodes and edges.
    cout << "node result: " << endl;
    for (auto const &node:f.first) {
        cout << node << " ";
    }
    cout << endl;
    cout << "edge result: " << endl;
    for (auto const &edge_index:f.second) {
        cout << edge_index << " ";
    }
    cout << endl;
}

int main() {
    cout << "test" << endl;
    test_tail_proj();
    test_head_proj();
}