# -*- coding: utf-8 -*-
__all__ = ["ghtp_logistic", "graph_ghtp_logistic"]
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


def ghtp_logistic(x_tr, y_tr, w0, lr, sparsity, tol, maximal_iter, eta):
    """
    :param x_tr: (n,p) training data
    :param y_tr: (n,) testing data
    :param w0: initial point, default is zero
    :param lr: learning rate
    :param sparsity: sparsity k
    :param tol: tolerance of algorithm for stop condition.
    :param maximal_iter: maximal number of iterations for GHTP algorithm.
    :param eta: regularization parameter eta
    :return: [losses, wt, intercept]
    """
    n_tr, p, wt = x_tr.shape[0], x_tr.shape[1], np.copy(w0)
    x_tr = np.concatenate((x_tr, np.ones(n_tr).reshape(n_tr, 1)), axis=1)
    losses = []
    for tt in range(num_iter):
        loss, grad = Algorithm._grad_w(x_tr, y_tr, wt, eta)
        w_tmp = wt - alpha * grad
        set_s = np.argsort(abs(w_tmp))[-s:]
        set_s = list(set_s)
        set_s.append(p)
        wt = Algorithm._min_f(set_s, x_tr, y_tr, wt, eta=eta)
        losses.append(loss)
        if len(losses) >= 2 and abs(losses[-2] - losses[-1]) < 1e-6:
            break


def graph_ghtp_logistic(x_tr, y_tr, sparsity, eta, tol, maximal_iter):
    pass
