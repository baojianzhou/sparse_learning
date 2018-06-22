
//
// Created by baojian on 6/21/18.
//
#include "union_find.h"

using std::cout;
using std::endl;
using std::cerr;
using std::pair;
using std::sort;
using std::vector;

void test_mst() {
    cout << "test" << endl;
    size_t n = 10;
    auto *unionFind = new UnionFind(n);
    unionFind->uf_union(0, 1);
    unionFind->uf_union(0, 2);
    unionFind->uf_union(0, 3);
    unionFind->uf_union(0, 4);
    unionFind->uf_union(0, 5);
    unionFind->uf_union(0, 6);
    cout << "number of components: " << unionFind->uf_num_cc() << endl;
    cout << "is connected: " << unionFind->uf_is_connected(3, 7) << endl;
    unionFind->uf_union(3, 7);
    cout << "is connected: " << unionFind->uf_is_connected(3, 7) << endl;
    unionFind->uf_union(7, 8);
    unionFind->uf_union(9, 7);
    cout << "number of components: " << unionFind->uf_num_cc() << endl;
    vector<pair<int, int>> edges;
    edges.emplace_back(0, 1);
    edges.emplace_back(0, 2);
    edges.emplace_back(0, 3);
    edges.emplace_back(0, 4);
    edges.emplace_back(0, 5);
    edges.emplace_back(0, 6);
    edges.emplace_back(3, 7);
    edges.emplace_back(7, 8);
    edges.emplace_back(9, 7);
    edges.emplace_back(8, 9);
    edges.emplace_back(4, 8);
    edges.emplace_back(2, 9);
    edges.emplace_back(2, 3);
    vector<double> weights;
    weights = {1., 2., .1, 1.2, 1.3, .5, .11, 11., 21., 31., 31., 0.1, 0.01};
    vector<size_t> selected_edges = kruskal_mst(edges, weights, 10);
    for (auto index : selected_edges) {
        cout << edges[index].first << " " << edges[index].second << endl;
    }
}

int main() {
    test_mst();
}