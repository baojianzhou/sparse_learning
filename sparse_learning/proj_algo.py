# -*- coding: utf-8 -*-
__all__ = ["HeadTailWrapper", "head_proj", "tail_proj"]
import numpy as np


class HeadTailWrapper(object):
    """
    The Python wrapper for the head and tail approx. algorithms.
    """

    def __init__(self, edges, weights):
        """ head and tail approximation package
        :param edges: ndarray[mx2] edges of the input graph
        :param weights: weights of edges
        """
        self._edges = edges
        self._weights = weights
        if not (self._weights > 0.0).all():
            print('Error: all edge weights must be positive.')
            exit()

    def run_tail(self, b, g, s, budget=None, nu=None):
        """ Run tail approximation algorithm
        :param b: input vector for projection.
        :param g: number of connected components
        :param s: sparsity
        :param budget: budget
        :param nu: parameter nu used in the tail approx. algorithm.
        :return: (nodes, edges,proj_vector):
        projected nodes, edges and projected vector.
        """
        if budget is None:
            budget = 1. * (s - g)
        if nu is None:
            nu = 2.5
        # if it is a zero vector, then just return an empty graph
        if not np.any(b):
            p_x = np.zeros_like(b)  # projected vector
            print('warning! tail input vector is a zero vector')
            return np.asarray([], dtype=int), np.asarray([], dtype=int), p_x
        return tail_proj(self._edges, self._weights, b, g, s, budget, nu)

    def run_head(self, b, g, s, budget=None, delta=None):
        """ Run head approximation algorithm.
        :param b: input vector for projection
        :param g:  number of connected component
        :param s: sparsity parameter
        :param budget: budget
        :param delta: parameter delta used in the head approx. algorithm.
        :return: (nodes, edges,proj_vector):
        projected nodes, edges and projected vector.
        """
        if budget is None:
            budget = 1. * (s - g)
        if delta is None:
            delta = 1. / 169.
        # if it is a zero vector, then just return an empty graph
        if not np.any(b):
            p_x = np.zeros_like(b)  # projected vector
            print('warning! head input vector is a zero vector')
            return np.asarray([], dtype=int), np.asarray([], dtype=int), p_x
        return head_proj(self._edges, self._weights, b, g, s, budget, delta)


def head_proj(edges, weights, b, g, s, budget, delta):
    """
    Head projection algorithm.
    :param edges: ndarray[mx2] edges of the input graph
    :param weights:  weights of edges
    :param b: input vector for projection
    :param g: number of connected component
    :param s: sparsity parameter
    :param budget:
    :param delta:
    :return:
    """
    re = [], []
    p_x = np.zeros_like(b)  # projected vector
    re_nodes, re_edges = re
    p_x[re_nodes] = b[re_nodes]
    return re_nodes, re_edges, p_x


def tail_proj(edges, weights, b, g, s, budget, nu):
    """
    Tail projection algorithm.
    :param edges: ndarray[mx2] edges of the input graph
    :param weights: weights of edges
    :param b: input vector for projection
    :param g: number of connected component
    :param s: sparsity parameter
    :param budget:
    :param nu:
    :return:
    """
    re = [], []
    p_x = np.zeros_like(b)  # projected vector
    re_nodes, re_edges = re
    p_x[re_nodes] = b[re_nodes]
    return re_nodes, re_edges, p_x
