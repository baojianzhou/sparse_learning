//
// Created by baojian on 6/29/18.
//

#ifndef SPARSE_PROJ_LOSS_FUNC_H
#define SPARSE_PROJ_LOSS_FUNC_H

/**
 * Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``
 * @param x (n_samples, n_features) dimension.
 * @param out log(1 / (1 + e ** -x[i]))
 * @param x_len
 */
void loss_logistic_sigmoid(const double *x,
                           double *out,
                           int x_len);


/**
 * Computes the logistic loss and gradient.
 * Parameters
 * ----------
 * @param w: (n_features,) or (n_features + 1,) Coefficient vector.
 * @param x: (n_samples, n_features)  Training data. (CblasRowMajor)
 * @param y: (n_samples,)           Array of labels.
 * @param alpha:   float Regularization parameter. equal to 1 / C.
 * @param weight: (n_samples,) optional
 *          Array of weights that are assigned to individual samples.
 *          If not provided, then each sample is given unit weight.
 * @param n_samples: number of samples
 * @param n_features: number of features
 * @return (loss, grad) (1,(n_features,)) or (1,(n_features + 1,)) loss, grad
 */
double *loss_logistic_loss_grad(const double *w,
                                const double *x,
                                const double *y,
                                double alpha,
                                const double *weight,
                                int w_len,
                                int n_samples,
                                int n_features);

/**
 * Computes the logistic loss and gradient.
 * Parameters
 * ----------
 * @param w: (n_features,) or (n_features + 1,) Coefficient vector.
 * @param x: (n_samples, n_features)            Training data.
 * @param y: (n_samples,)                       Array of labels.
 * @param alpha: float Regularization parameter.             equal to 1 / C.
 * @param weight: (n_samples,) optional
 *          Array of weights that are assigned to individual samples.
 *          If not provided, then each sample is given unit weight.
 * @param n_samples: number of samples
 * @param n_features: number of features
 * @return (loss) float Logistic loss.
 */
double loss_logistic_loss(const double *w,
                          const double *x,
                          const double *y,
                          double alpha,
                          const double *weight,
                          int w_len,
                          int n_samples,
                          int n_features);

double *loss_logistic_grad_hess(const double *w,
                                const double *x,
                                const double *y,
                                double alpha,
                                const double *weight,
                                int w_len,
                                int n_samples,
                                int n_features);

#endif //SPARSE_PROJ_LOSS_FUNC_H
