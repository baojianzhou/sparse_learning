# -*- coding: utf-8 -*-
__all__ = ['fast_pcst']
import numpy as np

try:
    from proj_module import proj_pcst
except ImportError:
    print('cannot find this functions: proj_pcst')
    exit(0)


def fast_pcst(edges, prizes, weights, root, g, pruning, verbose_level):
    """
    Fast PCST algorithm using C11 language
    :param edges:
    :param prizes:
    :param root:
    :param weights:
    :param g:
    :param pruning:
    :param verbose_level:
    :return:
    """
    if not np.any(prizes):  # make sure
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    if not (weights > 0.).all():
        print('all weights must be positive.')
    # TODO to check variables.
    re_nodes, re_edges = proj_pcst(edges, prizes, weights, root, g,
                                   pruning, verbose_level)
    print(re_nodes)
    print(re_edges)
    return re_nodes, re_edges
