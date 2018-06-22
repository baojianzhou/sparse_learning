/**
 *
 * This is a union-find set code to find a minimal spanning tree.
 *
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using std::cout;
using std::endl;
using std::cerr;
using std::pair;
using std::sort;
using std::vector;

class UnionFind {
public:
    explicit UnionFind(size_t n_) {
        num_connected_component = n_;
        size.resize(n_);
        parent.resize(n_);
        for (int i = 0; i < n_; i++) {
            parent[i] = i;
            size[i] = 1;
        }
    }

    int uf_num_cc() {
        return (int) num_connected_component;
    }

    int uf_find(int p) {
        if (check_valid(p)) {
            while (p != parent[p]) {
                p = parent[p];
            }
            return p;
        }
    }

    bool uf_is_connected(int p, int q) {
        return uf_find(p) == uf_find(q);
    }

    void uf_union(int p, int q) {
        int root_p = uf_find(p);
        int root_q = uf_find(q);
        if (root_p == root_q) {
            return;
        }
        if (size[root_p] < size[root_q]) {
            parent[root_p] = root_q;
            size[root_q] += size[root_p];
        } else {
            parent[root_q] = root_p;
            size[root_p] += size[root_q];
        }
        num_connected_component--;
    }

    ~UnionFind() = default;;

private:
    vector<int> size;
    vector<int> parent;
    size_t num_connected_component;

    bool check_valid(int p) {
        if (p < 0 || p >= parent.size()) {
            cerr << "index " << p << " is not between 0 and "
                 << (parent.size() - 1) << endl;
            return false;
        }
        return true;
    }

};

template<typename T>
vector<size_t> sort_indexes(const vector<T> &v) {
    vector<size_t> indices(v.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&v](size_t i1, size_t i2) {
        return v[i1] < v[i2];
    });
    return indices;
}

vector<size_t> kruskal_mst(const vector<pair<int, int>> &edges,
                           const vector<double> &weights, size_t num_nodes) {
    auto *unionFind = new UnionFind(num_nodes);
    vector<size_t> sorted_indices = sort_indexes(weights);
    vector<size_t> selected_edges;
    for (auto index:sorted_indices) {
        int u = edges[index].first;
        int v = edges[index].second;
        if (unionFind->uf_find(u) != unionFind->uf_find(v)) {
            selected_edges.push_back(index);
            unionFind->uf_union(u, v);
        }
    }
    return selected_edges;
}

