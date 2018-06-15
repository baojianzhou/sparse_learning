/*======================================================================
 * These two head and tail oracles are algorithms proposed in the paper:
 * Authors: Hegde, Chinmay, Piotr Indyk, and Ludwig Schmidt. (ICML 2015)
 * Title:   A nearly-linear time framework for graph-structured sparsity
 * Created by: Baojian Zhou, Email: bzhou6@albany.edu date: 05/25/2017
======================================================================*/
#ifndef HEAD_TAIL_H
#define HEAD_TAIL_H

#include <vector>

using std::min;
using std::pair;
using std::tuple;
using std::vector;


class HeadApprox {

public:

    //(edges_,prizes_,costs_) consist a graph
    HeadApprox(const vector<pair<int, int> > &edges_,
               const vector<double> &costs_,
               const vector<double> &prizes_,
               int g_, int s_, double C_, double delta_);

    // given an input vector b, return a forest F
    pair<vector<int>, vector<int> > run();

    // the nodes in the result forest.
    vector<int> result_nodes;
    // the edge indices in the result forest.
    vector<int> result_edges;

    ~HeadApprox();


private:

    const vector<pair<int, int> > &edges;
    const vector<double> &costs;
    const vector<double> &prizes;
    const int g;
    const int s;
    const double C;
    const double delta;
    const int verbose_level;

    // prizes could be changed, when call run(b) method.

    // sum of prizes.
    double total_prizes;
    // a tree is a pair, composed by lists of (first)nodes and (second)edges.
    typedef pair<vector<int>, vector<int> > Tree;
    typedef pair<vector<int>, vector<int> > Forest;

    // get positive minimum prize in prizes vector.
    double min_pi();

    // calculate pi(i) for each node i.
    vector<double> pi_lambda(double lambda);

    // calculate pi(F)
    double pi_f(Forest f);

    // calculate c(F)
    double cost_f(Forest f);

    // run fast-pcst algorithm.
    pair<vector<int>, vector<int> > pcsf_gw(vector<double> prizes);


    // prune a forest.
    pair<vector<int>, vector<int> > prune_forest(Forest f);

    // argmax tree is to create a single node tree
    Tree argmax_tree(Tree tree);

    vector<HeadApprox::Tree> sort_forest(Forest f);


    // prune a tree from original tree.
    Tree prune_tree(Tree t, double c_prime);


    // greater struct is for sort in descending order.
    struct greater {
        template<class T>
        bool operator()(T const &a, T const &b) const { return a > b; }
    };

    vector<int> get_nodes_map(vector<int> nodes);

    pair<vector<int>, vector<int> > dfs_tour(vector<int> nodes_t,
                                             vector<int> edges_t);

};

class TailApprox {

public:

    //(edges_,prizes_,costs_) consist a graph
    TailApprox(const vector<pair<int, int> > &edges_,
               const vector<double> &costs_,
               const vector<double> &prizes_,
               int g_, int s_, double C_, double nu_, double delta_);

    // given an input vector b, return a forest F
    pair<vector<int>, vector<int> > run();

    // the nodes in the result forest.
    vector<int> result_nodes;
    // the edge indices in the result forest.
    vector<int> result_edges;

private:
    const vector<pair<int, int> > &edges;
    const vector<double> &costs;
    const vector<double> &prizes;
    const int g;
    const int s;
    const double C;
    const double nu;
    const double delta;
    const int verbose_level;

    // sum of prizes.
    double total_prizes;

    // get positive minimum prize in prizes vector.
    double min_pi();

    // calculate c(e)*lambda_ for each edge.
    vector<double> cost_lambda(double lambda_);

    // calculate c(F).
    double cost_F(vector<int> result_edges);

    // calculate pi(\bar{F})
    double pi_F_bar(std::vector<int> result_nodes);

    // run fast-pcst algorithm.
    bool pcsf_gw(const vector<double> &costs);

};

#endif //HEAD_TAIL_H
