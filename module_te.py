#!/usr/bin/python
import numpy as np
from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj
from sparse_learning.fast_pcst import fast_pcst
from sparse_learning.proj_algo import HeadTailWrapper


def simu_graph(num_nodes):
    edges_, weights_ = [], []
    length = int(np.sqrt(num_nodes))
    width_, index_ = length, 0
    for i in range(length):
        for j in range(width_):
            if (index_ % length) != (length - 1):
                edges_.append((index_, index_ + 1))
                weights_.append(1.0)
                if index_ + length < int(width_ * length):
                    edges_.append((index_, index_ + length))
                    weights_.append(1.0)
            else:
                if index_ + length < int(width_ * length):
                    edges_.append((index_, index_ + length))
                    weights_.append(1.0)
            index_ += 1
    edges = np.asarray(edges_, dtype=int)
    weights = np.asarray(weights_, dtype=np.float64)
    return edges, weights


def test_all():
    edges, weights = simu_graph(25)  # get grid graph
    sub_graph = [6, 7, 8, 9]
    x = np.random.normal(0.0, 0.1, 25)
    x[sub_graph] = 5.
    # two options, you can use class or you can call function directly.
    # edges, weights, x, g, s, root, max_iter, budget, delta
    print('original head projection')
    re = head_proj(edges=edges, weights=weights, x=x, g=1, s=4, root=-1,
                   max_iter=20, budget=3., delta=1. / 169.)
    re_nodes, re_edges, p_x = re
    print('result head nodes: ', re_nodes)
    print('result head edges: ', re_edges)
    print('original tail projection')
    re = tail_proj(edges=edges, weights=weights, x=x, g=1, s=4, root=-1,
                   max_iter=20, budget=3., nu=2.5)
    re_nodes, re_edges, p_x = re
    print('result tail nodes: ', re_nodes)
    print('result tail edges: ', re_nodes)

    wrapper = HeadTailWrapper(edges=edges, weights=weights)
    re = wrapper.run_head(x=x, g=1, s=4, root=-1,
                          max_iter=20, budget=3, delta=1. / 169.)
    re_nodes, re_edges, p_x = re
    print('result head nodes: ', re_nodes)
    print('result head edges: ', re_nodes)
    re = wrapper.run_tail(x=x, g=1, s=4, root=-1,
                          max_iter=20, budget=3, nu=2.5)
    re_nodes, re_edges, p_x = re
    print('result tail nodes: ', re_nodes)
    print('result tail edges: ', re_nodes)
    # edges, weights, prizes, root, g, pruning, verbose_level
    re = fast_pcst(edges=edges, weights=weights, prizes=x ** 2.,
                   root=-1, g=1, pruning='strong', verbose_level=1)
    re_nodes, re_edges = re
    print('result pcst nodes: ', re_nodes)
    print('result pcst edges: ', re_nodes)


def main():
    test_all()


if __name__ == '__main__':
    main()
