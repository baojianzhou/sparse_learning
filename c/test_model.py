"""
=====================
Lasso and Elastic Net
=====================
Lasso and elastic net (L1 and L2 penalisation) implemented using a
coordinate descent.
The coefficients can be forced to be positive.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause
from head_tail import test
import time
import numpy as np
from itertools import cycle
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, enet_path


def compare_two_methods():
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)
    # Compute paths
    eps = 5e-3  # the smaller it is the longer is the path
    start_time = time.time()
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)
    run_time = time.time() - start_time
    alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
        X, y, eps, positive=True, fit_intercept=False)
    alphas_enet, coefs_enet, _ = enet_path(
        X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)
    alphas_positive_enet, coefs_positive_enet, _ = enet_path(
        X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)
    return alphas_lasso, alphas_enet, coefs_lasso, coefs_enet, \
           alphas_positive_lasso, alphas_positive_enet, \
           coefs_positive_lasso, coefs_positive_enet


def display(alphas_lasso, alphas_enet,
            coefs_lasso, coefs_enet,
            alphas_positive_lasso, alphas_positive_enet,
            coefs_positive_lasso, coefs_positive_enet):
    # Display results
    fig, ax = plt.subplots(2, 3)
    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    neg_log_alphas_enet = -np.log10(alphas_enet)
    for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
        l1 = ax[0, 0].plot(neg_log_alphas_lasso, coef_l, c=c)
        l2 = ax[0, 0].plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)
    ax[0, 0].set_xlabel('-Log(alpha)')
    ax[0, 0].set_ylabel('coefficients')
    ax[0, 0].set_title('Lasso and Elastic-Net Paths')
    ax[0, 0].legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
    neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)
    for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
        l1 = ax[0, 1].plot(neg_log_alphas_lasso, coef_l, c=c)
        l2 = ax[0, 1].plot(neg_log_alphas_positive_lasso, coef_pl, linestyle='--', c=c)
    ax[0, 1].set_xlabel('-Log(alpha)')
    ax[0, 1].set_ylabel('coefficients')
    ax[0, 1].set_title('Lasso and positive Lasso')
    ax[0, 1].legend((l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
    neg_log_alphas_positive_enet = -np.log10(alphas_positive_enet)
    for (coef_e, coef_pe, c) in zip(coefs_enet, coefs_positive_enet, colors):
        l1 = ax[0, 2].plot(neg_log_alphas_enet, coef_e, c=c)
        l2 = ax[0, 2].plot(neg_log_alphas_positive_enet, coef_pe, linestyle='--', c=c)
    ax[0, 2].set_xlabel('-Log(alpha)')
    ax[0, 2].set_ylabel('coefficients')
    ax[0, 2].set_title('Elastic-Net and positive Elastic-Net')
    ax[0, 2].legend((l1[-1], l2[-1]), ('Elastic-Net', 'positive Elastic-Net'),
                    loc='lower left')
    plt.axis('tight')
    plt.show()


def test_model():
    n, p = 3, 4
    mat = np.arange(12).reshape(n, p)
    re = test.test(n, p, mat)
    pass


def main():
    alphas_lasso, alphas_enet, coefs_lasso, coefs_enet, \
    alphas_positive_lasso, alphas_positive_enet, \
    coefs_positive_lasso, coefs_positive_enet = compare_two_methods()
    display(alphas_lasso, alphas_enet, coefs_lasso, coefs_enet,
            alphas_positive_lasso, alphas_positive_enet,
            coefs_positive_lasso, coefs_positive_enet)


if __name__ == '__main__':
    test_model()
    # main()
