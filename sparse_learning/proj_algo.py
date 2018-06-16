# -*- coding: utf-8 -*-
__all__ = ["HeadTailWrapper", "head_proj", "tail_proj"]
import numpy as np

try:
    import proj_module

    try:
        from proj_module import proj_head
        from proj_module import proj_tail
    except ImportError:
        print('cannot find these two functions: proj_head, proj_tail')
        exit(0)
except ImportError:
    print('cannot find the package proj_head')


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

    def run_tail(self, x, g, s, budget=None, nu=None, max_iter=None,
                 err_tol=None, root=None, pruning=None, verbose=None,
                 epsilon=None):
        """ Run tail approximation algorithm
        :param x: input vector for projection.
        :param g: number of connected components
        :param s: sparsity
        :param budget: budget
        :param nu: parameter nu used in the tail approx. algorithm.
        :param root:
        :param max_iter::
        :param err_tol
        :param epsilon:
        :param verbose:
        :param pruning:
        :return: (nodes, edges,proj_vector):
        projected nodes, edges and projected vector.
        """
        if budget is None:
            budget = 1. * (s - g)
        if nu is None:
            nu = 2.5
        if root is None:
            root = -1
        if max_iter is None:
            max_iter = 20
        if pruning is None:
            pruning = 'gw'
        if verbose is None:
            verbose = 0
        if epsilon is None:
            epsilon = 1e-6
        # if it is a zero vector, then just return an empty graph
        if not np.any(x):
            p_x = np.zeros_like(x)  # projected vector
            print('warning! tail input vector is a zero vector')
            return np.asarray([], dtype=int), np.asarray([], dtype=int), p_x
        return tail_proj(self._edges, self._weights, x, g, s, budget, nu,
                         max_iter, err_tol, root, pruning, epsilon, verbose)

    def run_head(self, x, g, s, budget=None, delta=None, max_iter=None,
                 err_tol=None, root=None, pruning=None, verbose=None,
                 epsilon=None):
        """ Run head approximation algorithm.
        :param x: input vector for projection
        :param g:  number of connected component
        :param s: sparsity parameter
        :param budget: budget
        :param delta: parameter delta used in the head approx. algorithm.
        :param root:
        :param max_iter
        :param err_tol
        :param epsilon:
        :param verbose:
        :param pruning:
        :return: (nodes, edges,proj_vector):
        projected nodes, edges and projected vector.
        """
        if budget is None:
            budget = 1. * (s - g)
        if delta is None:
            delta = 1. / 169.
        if root is None:
            root = -1
        if max_iter is None:
            max_iter = 20
        if pruning is None:
            pruning = 'gw'
        if verbose is None:
            verbose = 0
        if epsilon is None:
            epsilon = 1e-6
        # if it is a zero vector, then just return an empty graph
        if not np.any(x):
            p_x = np.zeros_like(x)  # projected vector
            print('warning! head input vector is a zero vector')
            return np.asarray([], dtype=int), np.asarray([], dtype=int), p_x
        return head_proj(self._edges, self._weights, x, g, s, budget, delta,
                         max_iter, err_tol, root, pruning, epsilon, verbose)


def head_proj(edges, weights, x, g, s, budget, delta, max_iter, err_tol,
              root, pruning, epsilon, verbose):
    """
    Head projection algorithm.
    :param edges: ndarray[mx2] edges of the input graph
    :param weights:  weights of edges
    :param x: input vector for projection
    :param g: number of connected component
    :param s: sparsity parameter
    :param budget:
    :param delta:
    :param max_iter: maximal iterations in head projection.
    :param err_tol: error tolerance for lower bound search bound.
    :param root: -1, no root for pcst
    :param pruning:
    :param epsilon:
    :param verbose:
    :return:
    """
    re_nodes, re_edges, p_x = proj_head(
        edges, weights, x, g, s, budget, delta, max_iter, err_tol,
        root, pruning, epsilon, verbose)
    print(re_nodes)
    print(re_edges)
    print(p_x)
    return re_nodes, re_edges, p_x


def tail_proj(edges, weights, x, g, s, budget, nu, max_iter, err_tol,
              root, pruning, verbose, epsilon):
    """
    Tail projection algorithm.
    :param edges: ndarray[mx2] edges of the input graph
    :param weights: weights of edges
    :param x: input vector for projection
    :param g: number of connected component
    :param s: sparsity parameter
    :param budget:
    :param nu:
    :param max_iter: maximal iterations
    :param err_tol:
    :param root: -1, no root for pcst
    :param pruning
    :param verbose
    :param epsilon
    :return:
    """
    # edges, weights, x, g, s, root, max_iter, budget, nu
    re_nodes, re_edges, p_x = proj_tail(
        edges, weights, x, g, s, budget, nu, max_iter, err_tol,
        root, pruning, epsilon, verbose)
    print(re_nodes)
    print(re_edges)
    print(p_x)
    return re_nodes, re_edges, p_x
