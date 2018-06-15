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
    b = np.random.normal(0.0, 0.1, 25)
    b[sub_graph] = 5.
    # two options, you can use class or you can call function directly.
    re = head_proj(edges=edges, weights=weights, x=b, g=1, s=4,
                   budget=3., delta=1. / 169.)
    nodes, edges, p_x = re
    print('result head nodes: ', nodes)
    print('result head edges: ', nodes)
    re = tail_proj(edges=edges, weights=weights, x=b, g=1, s=4,
                   budget=3., nu=2.5)
    nodes, edges, p_x = re
    print('result tail nodes: ', nodes)
    print('result tail edges: ', nodes)

    wrapper = HeadTailWrapper(edges=edges, weights=weights)
    re = wrapper.run_head(x=b, g=1, s=4, budget=3, delta=1. / 169.)
    nodes, edges, p_x = re
    print('result head nodes: ', nodes)
    print('result head edges: ', nodes)
    re = wrapper.run_tail(x=b, g=1, s=4, budget=3, nu=2.5)
    nodes, edges, p_x = re
    print('result tail nodes: ', nodes)
    print('result tail edges: ', nodes)
    re = fast_pcst(edges=edges, weights=weights, root=-1, g=1)
    nodes, edges = re
    print('result pcst nodes: ', nodes)
    print('result pcst edges: ', nodes)


def main():
    test_all()


if __name__ == '__main__':
    main()
