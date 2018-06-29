//
// Created by baojian on 6/29/18.
//

#ifndef SPARSE_PROJ_LOSS_FUNC_H
#define SPARSE_PROJ_LOSS_FUNC_H

double loss_logistic_primal_loss(const double w_xi,
                                 const double yi,
                                 const double weight);

double loss_logistic_primal_derivative(const double w_xi,
                                       const double yi,
                                       const double weight);

double *loss_logistic_intercept_dot(const double *w,
                                    const double *x,
                                    const double *y,
                                    int w_len,
                                    int x_len,
                                    int y_len);

/**
 * Scikit-learn api
 * Computes the logistic loss and gradient.
 * Parameters
 * ----------
 * w:       ndarray, (n_features,) or (n_features + 1,) Coefficient vector.
 * x:       ndarray, (n_samples, n_features)            Training data.
 * y:       ndarray, (n_samples,)                       Array of labels.
 * alpha:   float Regularization parameter.             equal to 1 / C.
 * sample_weight: (n_samples,) optional
 *          Array of weights that are assigned to individual samples.
 *          If not provided, then each sample is given unit weight.
 * @param n_samples: number of samples
 * @param n_features: number of features
 * @return (loss, grad) (1,(n_features,)) or (1,(n_features + 1,)) loss, grad
 */
double *loss_logistic_loss_and_grad(const double *w,
                                    const double *x,
                                    const double *y,
                                    double alpha,
                                    const double *weight,
                                    int w_len,
                                    int n_samples,
                                    int n_features);

/**
 * Scikit-learn api
 * Computes the logistic loss and gradient.
 * Parameters
 * ----------
 * w:       ndarray, (n_features,) or (n_features + 1,) Coefficient vector.
 * x:       ndarray, (n_samples, n_features)            Training data.
 * y:       ndarray, (n_samples,)                       Array of labels.
 * alpha:   float Regularization parameter.             equal to 1 / C.
 * sample_weight: (n_samples,) optional
 *          Array of weights that are assigned to individual samples.
 *          If not provided, then each sample is given unit weight.
 * @param n_samples: number of samples
 * @param n_features: number of features
 * @return (loss) float Logistic loss.
 */
double *loss_logistic_loss(const double *w,
                           const double *x,
                           const double *y,
                           double alpha,
                           const double *weight,
                           int w_len,
                           int n_samples,
                           int n_features);

#endif //SPARSE_PROJ_LOSS_FUNC_H
