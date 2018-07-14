//
// Created by baojian on 5/26/18.
// gcc -Wall -o main lasso.c -lcblas
#include <time.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <sys/types.h>

#define RAND_R_MAX  0x7FFFFFFF

//need to add extern, otherwise it has an error.
extern inline double fmax(double x, double y) {
    if (x > y) {
        return x;
    }
    return y;
}

extern inline double fsign(double f) {
    if (f == 0.0) {
        return 0.0;
    } else if (f > 0.0) {
        return 1.0;
    } else {
        return -1.0;
    }
}

double abs_max(int n, double *a) {
    int i;
    double m = fabs(a[0]);
    double d;
    for (i = 1; i < n; i++) {
        d = fabs(a[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

double max(int n, const double *a) {
    int i;
    double m = a[0];
    double d;
    for (i = 0; i < n; i++) {
        d = a[i];
        if (d > m) {
            m = d;
        }
    }
    return m;
}


double diff_abs_max(int n, double *a, double *b) {
    int i;
    double m = fabs(a[0] - b[0]);
    double d;
    for (i = 1; i < n; i++) {
        d = fabs(a[i] - b[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

extern inline size_t rand_int(int end, u_int32_t *seed) {
    seed[0] ^= (u_int32_t) (seed[0] << 13);
    seed[0] ^= (u_int32_t) (seed[0] << 17);
    seed[0] ^= (u_int32_t) (seed[0] << 5);
    return (seed[0] % ((u_int32_t) RAND_R_MAX + 1)) % end;
}


struct Result_CD {
    double *w;
    double gap;
    double tol;
    long n_iter;
};

struct Result_ENet {
    double *alphas;
    double *coefs;
    double dual_gaps;
    int n_iters;
};

PyArrayObject *check_array(PyArrayObject *array,
                           void *accept_sparse,
                           char *dtype,
                           char *order,
                           int copy,
                           int force_all_finite,
                           int ensure_2d,
                           int allow_nd,
                           int ensure_min_samples,
                           int warn_on_dtype,
                           char *estimator) {
    return array;
}

int isspmatrix(PyArrayObject *x) {
    return 0;
}

int is_str_in(char *s, char *x[]) {
    int len = sizeof(x) / sizeof(x[0]);
    int i;
    for (i = 0; i < len; ++i) {
        if (!strcmp(x[i], s)) {
            return 1;
        }
    }
    return 0;
}

/*****************************************************************************
 *
 * Compute elastic net path with coordinate descent
 * The elastic net optimization function varies for mono and multi-outputs.
 * 1. For mono-output tasks:
 *      min_w   1/(2*n)*||y - Xw||^2_2
 *              + alpha*l1_ratio*||w||_1 + 0.5*alpha*(1 - l1_ratio)*||w||^2_2
 * 2. For multi-output tasks:
 *      min_w   (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
 *              + alpha * l1_ratio * ||W||_21
 *              + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2, where
 *              ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2},
 *              i.e., the sume of norm of each row
 *
 * @param x (n=n_samples, p=n_features). Training data. Pass directly as
 *          Fortran-contiguous data to avoid unnecessary memory duplication.
 *          If y is mono-output then X can be sparse.
 * @param y (n=n_samples), or (n=n_samples,k=n_outputs) Target values
 * @param l1_ratio float, [opt] float between 0 and 1 passed to elastic net
 *          (scaling between l1 and l2 penalties).
 *          ``l1_ratio=1`` corresponds to the Lasso
 * @param eps float, [opt] Length of the path. ``eps=1e-3`` means that
 *          ``alpha_min / alpha_max = 1e-3``
 * @param n_alphas int, [opt] Number of alphas along the regularization path
 * @param alphas [opt] List of alphas where to compute the models. If ``None``
 *          alphas are set automatically
 * @param precompute True | False | 'auto' | array-like Whether to use a
 *          precomputed Gram matrix to speed up calculations. If set to
 *          auto let us decide. The Gram matrix can also be passed as argument.
 * @param xy array-like, [opt] xy = np.dot(x.T, y) that can be precomputed.
 *          It is useful only when the Gram matrix is precomputed.
 * @param copy_x boolean, [opt] default True
 *          If ``True``, X will be copied; else, it may be overwritten.
 * @param coef_init array, shape (n_features, ) | None
 *          The initial values of the coefficients.
 * @param verbose bool or integer Amount of verbosity.
 * @param return_n_iter bool whether to return the number of iterations or not.
 * @param positive bool, default False. If set to True, forces coefficients
 *          to be positive. (Only allowed when ``y.ndim == 1``).
 * @param check_input bool, default True. Skip input validation checks,
 *          including the Gram matrix when provided assuming there are handled
 *          by the caller when check_input=False.
 * @param ... kwargs keyword arguments passed to the coordinate descent solver.
 *
 * @return
 *****************************************************************************/
struct Result_ENet enet_path(
        PyArrayObject *x,
        PyArrayObject *y,
        double l1_ratio,
        double eps,
        int n_alphas,
        PyArrayObject *alphas,
        int precompute,
        PyArrayObject *xy,
        int copy_x,
        PyArrayObject *coef_init,
        int verbose,
        int positive,
        int return_n_iter,
        int check_input,
        PyArrayObject *x_offset,
        PyArrayObject *x_scale, ...) {

    // we assume that users input -1 as default value.
    if (positive == -1) {
        positive = 0;
    }
    if (check_input == -1) {
        check_input = 1;
    }
    // x and y should be already Fortran ordered when bypassing
    if (check_input) {
        x = check_array(x, "csc", "float64, float32", "F", copy_x, -1,
                        -1, -1, -1, -1, " ");
        y = check_array(y, "csc", "float64, float32", "F", 0, 0,
                        -1, -1, -1, -1, " ");
        // Xy should be a 1d contiguous array or a 2D C ordered array
        if (xy != NULL) {
            xy = check_array(xy, " ", "float64, float32", "F", 0, 0, -1, -1,
                             -1, -1, " ");
        }
    }
    if (PyArray_NDIM(x) != 2) {
        printf("Error: dimension of x is inconsistent!");
        exit(0);
    } else {
        long int n = x->dimensions[0], p = x->dimensions[1];
    }
    int multi_output = 0;
    long int n_outputs = -1;
    double x_sparse_scaling = 0.0;
    if (PyArray_NDIM(y) != 1) {
        multi_output = 1;
        n_outputs = y->dimensions[1];
    }
    if (multi_output && positive) {
        printf("Error: positive=True is not allowed for multi-output");
        exit(0);
    }
    if (!multi_output && isspmatrix(x)) {
        // As sparse matrices are not actually centered we need this
        // to be passed to the CD solver.
        if (x_offset != NULL) {
            x_sparse_scaling = 0.0;
        } else {
            x_sparse_scaling = 0.0;
        }
    }


    struct Result_ENet result_enet = {NULL, NULL, 0., 0};
    return result_enet;
}

/*****************************************************************************
 *  Compute Lasso path with coordinate descent.
 *  The Lasso optimization function varies for mono and multi-outputs.
 *  1. For mono-output tasks it is:
 *          min_w (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
 *  2. For multi-output tasks it is:
 *          min_w (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21,
 *              where ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}
 *
 * @param x See         see @def enet_path
 * @param y See         see @def enet_path
 * @param eps           see @def enet_path
 * @param n_alphas      see @def enet_path
 * @param alphas        see @def enet_path
 * @param precompute    see @def enet_path
 * @param xy            see @def enet_path
 * @param copy_x        see @def enet_path
 * @param coef_init     see @def enet_path
 * @param verbose       see @def enet_path
 * @param return_n_iter see @def enet_path
 * @param positive      see @def enet_path
 * @return              see @def enet_path
 ****************************************************************************/
struct Result_ENet lasso_path(
        PyArrayObject *x,
        PyArrayObject *y,
        double eps,
        int n_alphas,
        PyArrayObject *alphas,
        int precompute,
        PyArrayObject *xy,
        int copy_x,
        PyArrayObject *coef_init,
        int verbose,
        int return_n_iter,
        int positive,
        PyArrayObject *x_offset,
        PyArrayObject *x_scale, ...) {

    return enet_path(x, y, 1., eps, n_alphas, alphas, precompute,
                     xy, copy_x, coef_init, verbose, positive,
                     return_n_iter, positive, x_offset, x_scale);
}


/*****************************************************************************
 * c version of the coordinate descent algorithm for elastic-net
 * to minimize:
 *    (1/2)* norm(y-Xw,2)^2 + alpha * norm(w,1) + (beta/2) * norm(w,2)^2
 *****************************************************************************/
struct Result_CD enet_coordinate_descent(
        double *w,
        double alpha,
        double beta,
        double *x,
        double *y,
        int max_iter,
        double tol,
        int random,
        int positive,
        int n_samples,
        int n_features) {

    size_t ii, i, n_iter = 0, f_iter; // get the number of tasks TODO
    const int n = n_samples, p = n_features, n_tasks = n_samples;
    double *norm_cols_x = (double *) malloc(sizeof(double) * p);
    for (i = 0; i < p; i++) {
        norm_cols_x[i] = cblas_ddot(n, x + i, p, x + i, p);
    }
    double *r = (double *) malloc(sizeof(double) * n); // value of residuals
    double *xta = (double *) malloc(sizeof(double) * p);
    double tmp, w_ii, d_w_max, w_max, d_w_ii, gap = tol + 1.0;
    double d_w_tol = tol, dual_norm_xta;
    double r_norm2, w_norm2, l1_norm, const_, a_norm2;
    srand(time(NULL)); // Notice that we need this.
    u_int32_t rand_r_state_seed = (u_int32_t) (rand() - 1); // [0,RAND_R_MAX)
    double *x_data = x, *y_data = y, *w_data = w, *r_data = r, *xta_data = xta;
    if ((alpha <= 0.0) && (beta <= 0.0)) {
        printf("Warning: Coordinate descent with no regularization \n"
               "may lead to unexpected results and is discouraged !\n");
    }
    for (i = 0; i < n; i++) {   //residual r = y - dot(X,w)
        r[i] = y[i] - cblas_ddot(p, &x_data[i], n, w_data, 1);
    }
    tol *= cblas_ddot(n, y_data, n_tasks, y_data, n_tasks); //tol *= dot(y,y)
    for (n_iter = 0; n_iter < max_iter; n_iter++) {
        w_max = 0.0, d_w_max = 0.0;
        for (f_iter = 0; f_iter < p; f_iter++) {
            if (random) {
                ii = rand_int(p, &rand_r_state_seed);
            } else {
                ii = f_iter;
            }
            if (norm_cols_x[ii] == 0.0) {
                continue;
            }
            w_ii = w[ii]; // save previous value.
            if (w_ii != 0.0) { // R += w_ii * X[:, ii]
                cblas_daxpy(n, w_ii, &x_data[ii * n], 1, r_data, 1);
            }
            tmp = cblas_ddot(n, &x_data[ii * n], 1, r_data, 1);
            if (positive && tmp < 0) {
                w[ii] = 0.0;
            } else {
                w[ii] = (fsign(tmp) * fmax(fabs(tmp) - alpha, 0) /
                         (norm_cols_x[ii] + beta));
            }
            if (w[ii] != 0.0) {
                cblas_daxpy(n, -w[ii], &x_data[ii * n], 1, r_data, 1);
            }
            // update the maximum absolute coefficient update
            d_w_ii = fabs(w[ii] - w_ii);
            if (d_w_ii > d_w_max) {
                d_w_max = d_w_ii;
            }
            if (fabs(w[ii]) > w_max) {
                w_max = fabs(w[ii]);
            }
        }
        int condi_1 = (w_max == 0.0);
        int condi_2 = (d_w_max / w_max < d_w_tol);
        int condi_3 = (n_iter == (max_iter - 1));
        if (condi_1 || condi_2 || condi_3) {
            // the biggest coordinate update of this iteration was smaller
            // than the tolerance: check the duality gap as ultimate
            // stopping criterion
            for (i = 0; i < p; i++) {
                xta[i] = cblas_ddot(n, &x_data[i * n], 1, r_data, 1) -
                         beta * w[i];
            }
            if (positive) {
                dual_norm_xta = max(p, xta_data);
            } else {
                dual_norm_xta = abs_max(p, xta_data);
            }
            r_norm2 = cblas_ddot(n, r_data, 1, r_data, 1);
            w_norm2 = cblas_ddot(n, w_data, 1, w_data, 1);
            if (dual_norm_xta > alpha) {
                const_ = alpha / dual_norm_xta;
                a_norm2 = r_norm2 * (const_ * const_);
                gap = 0.5 * (r_norm2 + a_norm2);
            } else {
                const_ = 1.0;
                gap = r_norm2;
            }
            l1_norm = cblas_dasum(p, w_data, 1);
            gap += (alpha * l1_norm)
                   - const_ * cblas_ddot(n, r_data, 1, y_data, n_tasks)
                   + 0.5 * beta * (1 + const_ * const_) * (w_norm2);
            if (gap < tol) {
                break; // return if we reached the desired tolerance.
            }
        }
    }
    free(norm_cols_x); // release memory
    free(r);
    free(xta);
    struct Result_CD result = {w, gap, tol, n_iter + 1};
    return result;
}


/*****************************************************************************
 * The following part is the for Python of C extension.
 ****************************************************************************/


static PyObject *test(PyObject *self, PyObject *args) {
    npy_int64 n;
    npy_float64 p;
    PyArrayObject *py_m;
    // l:long d: double, O!: an object "ldO!O!"
    if (!PyArg_ParseTuple(args, "ldO!",
                          &n,
                          &p,
                          &PyArray_Type, &py_m)) {
        printf("something wrong");
    }
    printf("n: %ld\n", n);
    printf("p: %lf\n", p);
    printf("size: %d\n", (*py_m).nd);
    printf("n: %ld\n", (*py_m).dimensions[0]);
    printf("p: %ld\n", (*py_m).dimensions[1]);
    long int len = PyArray_Size((PyObject *) py_m);
    printf("size: %ld\n", (len));
    int i = 0, j = 0, index = 0;
    for (i = 0; i < py_m->dimensions[0]; i++) {
        for (j = 0; j < py_m->dimensions[1]; j++) {
            long int *element = (long int *) PyArray_GETPTR2(py_m, i, j);
            printf(" %ld", *element);
            index++;
        }
        printf("\n");
    }
    // build results
    const Py_ssize_t tuple_length = 4;
    const unsigned some_limit = 4;

    PyObject *my_list = PyList_New(0);
    if (my_list == NULL) {
        // ...
    }
    for (i = 0; i < some_limit; i++) {
        PyObject *the_tuple = PyTuple_New(tuple_length);
        if (the_tuple == NULL) {
            // ...
        }
        for (j = 0; i < tuple_length; i++) {
            PyObject *the_object = PyLong_FromSsize_t(i * tuple_length + j);
            if (the_object == NULL) {
                // ...
            }
            PyTuple_SET_ITEM(the_tuple, j, the_object);
        }
        if (PyList_Append(my_list, the_tuple) == -1) {
            // ...
        }
    }
    return my_list;
}

//gcc -Wall -o main lasso.c -lcblas
int main() {
    printf("test\n");
}

/*  define functions in module */
static PyMethodDef CosMethods[] = {
        {"test", test, METH_VARARGS, "evaluate the cosine"},
        {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,
        "test", "Some documentation",
        -1,
        CosMethods
};

PyMODINIT_FUNC PyInit_cos_module(void) {
    return PyModule_Create(&cModPyDem);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC inittest(void) {
    import_array();
    (void) Py_InitModule("test", CosMethods);
}

#endif