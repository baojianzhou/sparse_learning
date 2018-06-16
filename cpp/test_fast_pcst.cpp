/**=========================================================================
 * The fast-pcst algorithm, including pairing_heap.h, pcst_fast.cc,
 * pcst_fast.h and priority_queue.h was implemented by Ludwig Schmidt.
 * You can check his original code at the following two urls:
 *      https://github.com/ludwigschmidt/pcst-fast
 *      https://github.com/fraenkel-lab/pcst_fast.git

MIT License

Copyright (c) 2017 Ludwig Schmidt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===========================================================================*/
#include <iostream>
#include <algorithm>
#include <utility>
#include <vector>
#include "pcst_fast.h"

using std::make_pair;
using std::pair;
using std::vector;
using std::cout;
using std::endl;
using cluster_approx::PCSTFast;

void print_test(vector<int> node_result, vector<int> exp_node_result,
                vector<int> edge_result, vector<int> exp_edge_result) {
    cout << "-------------------------------------" << endl;
    cout << "node results: " << endl;
    for (int node:node_result) {
        cout << node << " ";
    }
    cout << endl << "expected node results: " << endl;
    for (int node:exp_node_result) {
        cout << node << " ";
    }
    cout << endl << "edge results: " << endl;
    for (int node:edge_result) {
        cout << node << " ";
    }
    cout << endl << "expected edge results: " << endl;
    for (int node:exp_edge_result) {
        cout << node << " ";
    }
    cout << endl;
}

void test_1() {
    vector<pair<int, int> > edges;
    edges.push_back(make_pair(0, 1));
    edges.push_back(make_pair(1, 2));
    vector<double> prizes = {0, 5, 6};
    vector<double> costs = {3, 4};
    int root = 0;
    int target_num_active_clusters = 0;
    PCSTFast::PruningMethod pruning = PCSTFast::kNoPruning;

    vector<int> exp_node_result = {0, 1, 2};
    vector<int> exp_edge_result = {0, 1};
    vector<int> node_result;
    vector<int> edge_result;

    PCSTFast algo(edges, prizes, costs, root, target_num_active_clusters,
                  pruning, 0, NULL);
    algo.run(&node_result, &edge_result);
    std::sort(node_result.begin(), node_result.end());
    std::sort(edge_result.begin(), edge_result.end());
    vector<int> sorted_expected_node_result(exp_node_result);
    std::sort(sorted_expected_node_result.begin(),
              sorted_expected_node_result.end());
    vector<int> sorted_expected_edge_result(exp_edge_result);
    std::sort(sorted_expected_edge_result.begin(),
              sorted_expected_edge_result.end());
    print_test(node_result, exp_node_result, edge_result, exp_edge_result);
}

void test_2() {
    vector<pair<int, int> > edges;
    edges.push_back(make_pair(0, 1));
    edges.push_back(make_pair(1, 2));
    vector<double> prizes = {0, 5, 6};
    vector<double> costs = {3, 4};
    int root = -1;
    int target_num_active_clusters = 1;
    PCSTFast::PruningMethod pruning = PCSTFast::kNoPruning;

    vector<int> exp_node_result = {1, 2};
    vector<int> exp_edge_result = {1};
    vector<int> node_result;
    vector<int> edge_result;

    PCSTFast algo(edges, prizes, costs, root, target_num_active_clusters,
                  pruning, 0, NULL);
    algo.run(&node_result, &edge_result);
    std::sort(node_result.begin(), node_result.end());
    std::sort(edge_result.begin(), edge_result.end());
    vector<int> sorted_expected_node_result(exp_node_result);
    std::sort(sorted_expected_node_result.begin(),
              sorted_expected_node_result.end());
    vector<int> sorted_expected_edge_result(exp_edge_result);
    std::sort(sorted_expected_edge_result.begin(),
              sorted_expected_edge_result.end());
    print_test(node_result, exp_node_result, edge_result, exp_edge_result);
}

void test_3() {
    vector<pair<int, int> > edges;
    edges.push_back(make_pair(0, 1));
    edges.push_back(make_pair(1, 2));
    vector<double> prizes = {0, 5, 6};
    vector<double> costs = {3, 4};
    int root = -1;
    int target_num_active_clusters = 1;
    PCSTFast::PruningMethod pruning = PCSTFast::kGWPruning;

    vector<int> exp_node_result = {1, 2};
    vector<int> exp_edge_result = {1};
    vector<int> node_result;
    vector<int> edge_result;

    PCSTFast algo(edges, prizes, costs, root, target_num_active_clusters,
                  pruning, 0, NULL);
    algo.run(&node_result, &edge_result);
    std::sort(node_result.begin(), node_result.end());
    std::sort(edge_result.begin(), edge_result.end());
    vector<int> sorted_expected_node_result(exp_node_result);
    std::sort(sorted_expected_node_result.begin(),
              sorted_expected_node_result.end());
    vector<int> sorted_expected_edge_result(exp_edge_result);
    std::sort(sorted_expected_edge_result.begin(),
              sorted_expected_edge_result.end());
    print_test(node_result, exp_node_result, edge_result, exp_edge_result);
}

void test_4() {
    vector<pair<int, int> > edges;
    edges.push_back(make_pair(0, 1));
    edges.push_back(make_pair(1, 2));
    edges.push_back(make_pair(2, 3));
    edges.push_back(make_pair(3, 4));
    edges.push_back(make_pair(4, 5));
    edges.push_back(make_pair(5, 6));
    edges.push_back(make_pair(6, 7));
    vector<double> prizes = {100.0,
                             0.0,
                             0.0,
                             1.0,
                             0.0,
                             0.0,
                             0.0,
                             100.0};
    vector<double> costs = {0.9,
                            0.9,
                            0.9,
                            0.9,
                            0.9,
                            0.9,
                            0.9};
    int root = -1;
    int target_num_active_clusters = 1;
    PCSTFast::PruningMethod pruning = PCSTFast::kGWPruning;


    vector<int> exp_node_result = {0, 1, 2, 3, 4, 5, 6, 7};
    vector<int> exp_edge_result = {0, 1, 2, 3, 4, 5, 6};
    vector<int> node_result;
    vector<int> edge_result;

    PCSTFast algo(edges, prizes, costs, root, target_num_active_clusters,
                  pruning, 0, NULL);
    algo.run(&node_result, &edge_result);
    std::sort(node_result.begin(), node_result.end());
    std::sort(edge_result.begin(), edge_result.end());
    vector<int> sorted_expected_node_result(exp_node_result);
    std::sort(sorted_expected_node_result.begin(),
              sorted_expected_node_result.end());
    vector<int> sorted_expected_edge_result(exp_edge_result);
    std::sort(sorted_expected_edge_result.begin(),
              sorted_expected_edge_result.end());
    print_test(node_result, exp_node_result, edge_result, exp_edge_result);
}

void test_5() {
    vector<pair<int, int> > edges;
    edges.push_back(make_pair(0, 1));
    edges.push_back(make_pair(1, 2));
    edges.push_back(make_pair(2, 3));
    edges.push_back(make_pair(0, 9));
    edges.push_back(make_pair(0, 2));
    edges.push_back(make_pair(0, 3));
    edges.push_back(make_pair(0, 5));
    edges.push_back(make_pair(1, 9));
    edges.push_back(make_pair(1, 3));
    edges.push_back(make_pair(1, 5));
    edges.push_back(make_pair(1, 7));
    edges.push_back(make_pair(2, 8));
    edges.push_back(make_pair(2, 3));
    edges.push_back(make_pair(3, 4));
    edges.push_back(make_pair(3, 5));
    edges.push_back(make_pair(3, 6));
    edges.push_back(make_pair(3, 7));
    edges.push_back(make_pair(3, 8));
    edges.push_back(make_pair(3, 9));
    edges.push_back(make_pair(4, 5));
    edges.push_back(make_pair(4, 6));
    edges.push_back(make_pair(4, 7));
    edges.push_back(make_pair(5, 8));
    edges.push_back(make_pair(6, 8));
    vector<double> prizes = {0.032052554364677466,
                             0.32473378289799926,
                             0.069699345546302638,
                             0,
                             0.74867253235151754,
                             0.19804330340026255,
                             0.85430521133171622,
                             0.83819939651391351,
                             0.71744625276884877,
                             0.016798567754083948};
    vector<double> costs = {0.8,
                            0.8,
                            0.8800000000000001,
                            0.8,
                            0.8,
                            0.8800000000000001,
                            0.8,
                            0.8,
                            0.8800000000000001,
                            0.8,
                            0.8,
                            0.8,
                            0.8800000000000001,
                            0.8800000000000001,
                            0.8800000000000001,
                            0.8800000000000001,
                            0.8800000000000001,
                            0.8800000000000001,
                            0.8800000000000001,
                            0.8,
                            0.8,
                            0.8,
                            0.8,
                            0.8};
    int root = 3;
    int target_num_active_clusters = 0;
    PCSTFast::PruningMethod pruning = PCSTFast::kGWPruning;

    vector<int> exp_node_result = {3, 4, 6, 7, 8};
    vector<int> exp_edge_result = {16, 20, 21, 23};
    vector<int> node_result;
    vector<int> edge_result;
    PCSTFast algo(edges, prizes, costs, root, target_num_active_clusters,
                  pruning, 0, NULL);
    algo.run(&node_result, &edge_result);
    std::sort(node_result.begin(), node_result.end());
    std::sort(edge_result.begin(), edge_result.end());
    vector<int> sorted_expected_node_result(exp_node_result);
    std::sort(sorted_expected_node_result.begin(),
              sorted_expected_node_result.end());
    vector<int> sorted_expected_edge_result(exp_edge_result);
    std::sort(sorted_expected_edge_result.begin(),
              sorted_expected_edge_result.end());
    print_test(node_result, exp_node_result, edge_result, exp_edge_result);
}

int main() {
    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
}