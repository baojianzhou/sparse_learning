/****************************************************************************
 * This program is Online-Graph-GHTP implemented by Baojian Zhou
 * You need to have compile it by the following command:
 * cmake .. // under build sub-folder
 * make
 * It depends on the following two packages:
 * 1. Python2.7
 * 2. openblas or lapack
 ***************************************************************************/
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <Python.h>
#include <cblas.h>
#include <numpy/arrayobject.h>

typedef float f32;
typedef double f64;
typedef int Edge[2];

typedef struct Forest {
    int num_nodes;
    int num_edges;
    int *edges; // index edges
    int *nodes; // nodes
    double *proj_w;
} Forest;

typedef struct InputData {
    long int n; // number of samples
    long int p; // number of features
    long int size;
    unsigned int s_low; // lower bound sparsity
    unsigned int s_high; // upper bound sparsity
    double *x;  // n x p, data matrix
    double *y;     // n x 1, label vector
    double *w0; // p x 1, initial vector
    Edge *edges;
    double *edge_costs;
    double edge_costs_multiplier;
    int root;
    int num_active_clusters;
    char *pruning;
    char *pathway_id;
    int verbose_level;
    unsigned int max_num_iter;
} InputData;

/** to calculate 1/(1+exp(-x)) */
extern inline double logistic(double x) {
    if (x > 0.) {
        return (1. / (1. + exp(-x)));
    } else {
        double exp_x = exp(x);
        return (exp_x / (1. + exp_x));
    }
}

/** v := v ** 2 */
extern inline void self_product(long int p, double *v) {
    for (int i = 0; i < p; i++) {
        v[i] *= v[i];
    }
}

/** summation of a vector v. sum(v) */
extern inline double sum(long int p, const double *v) {
    double sum_ = 0.0;
    for (int i = 0; i < p; i++) {
        sum_ += v[i];
    }
    return sum_;
}

/** arg min_{v_i !=0} v_i */
extern inline double nonzero_min(int p, const double *v) {
    double _min = v[0];
    for (int i = 0; i < p; i++) {
        if (0 < v[i] < _min) {
            _min = v[i];
        }
    }
    // TODO need to check very small positive value. 1e-236 ?
    return _min;
}

/*****************************************************************************
 * Return logistic gradient and loss for one sample:
 * `\log(1+\exp(-y_i*(w^t*x_i))) + eta *\|w\|_2^2`
 * @param xi training data xi
 * @param yi training label yi `yi \in {+1,-1}`
 * @param w weight
 * @param eta eta > 0 as a regularization parameter
 * @return grad, loss: the last element is a loss
 ****************************************************************************/
void logistic_grad(const int p, const double *xi, const double yi,
                   const double *w, const double eta, double *grad_loss) {
    double x_dot = -yi * cblas_ddot(p, w, 1, xi, 1);
    double b = fmax(0.0, x_dot);
    grad_loss[p] = b + log(exp(-b) + exp(x_dot - b)); // loss
    cblas_dcopy(p, xi, 1, grad_loss, 1);
    cblas_dscal(p, -yi * logistic(x_dot), grad_loss, 1); // grad
    if (eta > 0) {
        grad_loss[p] += eta * cblas_ddot(p, w, 1, w, 1); // reg_loss
        double *reg_gradient = (double *) malloc(sizeof(double) * p);
        cblas_dcopy(p, w, 1, reg_gradient, 1);
        cblas_dscal(p, 2. * eta, reg_gradient, 1);
        cblas_daxpy(p, 1., reg_gradient, 1, grad_loss, 1);
        free(reg_gradient);
    }
}

/** logistic regression for n samples.*/
void batch_logistic_grad(const double *x, const double *y, const double *w,
                         int n, int p, double *grad_loss) {
    memset(grad_loss, 0, sizeof(double) * (p + 1)); // to zero.
    double *x_dot = (double *) malloc(sizeof(double) * n);
    double *tmp_grad_loss = (double *) malloc(sizeof(double) * (p + 1));
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                n, p, 1., x, n, w, 1, 0., x_dot, 1);
    for (int i = 0; i < n; i++) {
        double b = fmax(0.0, x_dot[i]);
        grad_loss[p] += b + log(exp(-b) + exp(x_dot[i] - b)); // loss
        cblas_dcopy(p, x + i * p, 1, tmp_grad_loss, 1);
        cblas_dscal(p, -y[i] * logistic(x_dot[i]), tmp_grad_loss, 1);
        cblas_daxpy(p, 1., tmp_grad_loss, 1, grad_loss, 1);
    }
    free(tmp_grad_loss);
}

Forest *fast_pcst(Edge *edges, double *prizes, double *costs,
                  int g, char *pruning, int root, int p, int verbose_level) {
    Forest *results = (Forest *) malloc(sizeof(Forest));
    results->edges = NULL;
    results->nodes = NULL;
    return results;
}


/****************************************************************************
 *
 * @param w
 * @param input_data
 * @return
 ***************************************************************************/
Forest *binary_search_tail(const double *w, const InputData *input_data) {
    int p = (int) input_data->p, s_low = input_data->s_low;
    double *prizes = (double *) malloc(sizeof(double) * p);
    cblas_dcopy(p, w, 1, prizes, 1);
    self_product(p, prizes);
    double *proj_w = (double *) calloc(sizeof(double), (size_t) p);
    if (sum(p, prizes) <= 0.0) {
        printf("warning: zero vector !");
        Forest *re_f = (Forest *) malloc(sizeof(Forest));
        re_f->num_nodes = 0, re_f->num_edges = 0; // empty tree
        re_f->proj_w = proj_w;
        free(prizes);
        return re_f; // in this case, just return a zero vector.
    }
    double *costs = (double *) malloc(sizeof(double) * p);
    cblas_dcopy(p, input_data->edge_costs, 1, costs, 1);
    cblas_dscal(p, input_data->edge_costs_multiplier, costs, 1);
    double lambda_r = 0.0, lambda_l = 3. * cblas_dasum(p, prizes, 1);
    double eps = 0.01 * nonzero_min(p, prizes), lambda_m;
    int num_iter = 0, cur_sparsity, i, max_num_it = input_data->max_num_iter;
    if (input_data->verbose_level >= 2) {
        printf("Initial lambda_l:%.6f lambda_r:%.6f eps: %.6f\n",
               lambda_l, lambda_r, eps);
    }
    while (((lambda_l - lambda_r) > eps) && (num_iter < max_num_it)) {
        num_iter++;
        lambda_m = (lambda_l + lambda_r) / 2.;
        cblas_dcopy(p, input_data->edge_costs, 1, costs, 1);
        cblas_dscal(p, lambda_m, costs, 1);
        Forest *f = fast_pcst(input_data->edges, prizes, costs,
                              input_data->num_active_clusters,
                              input_data->pruning, input_data->root, p,
                              input_data->verbose_level - 1);
        cur_sparsity = f->num_nodes;
        if (input_data->s_low <= cur_sparsity <= input_data->s_high) {
            for (i = 0; i < f->num_nodes; i++) {
                proj_w[i] = w[f->nodes[i]];
            }
            f->proj_w = proj_w;
            free(costs);
            free(prizes);
            return f;
        } else {
            free(f);
        }
        if (cur_sparsity > input_data->s_high) {
            lambda_r = lambda_m;
        } else {
            lambda_l = lambda_m;
        }
    } // while
    cblas_dcopy(p, input_data->edge_costs, 1, costs, 1);
    cblas_dscal(p, lambda_l, costs, 1);
    Forest *f = fast_pcst(input_data->edges, prizes, costs,
                          input_data->num_active_clusters,
                          input_data->pruning, input_data->root, p,
                          input_data->verbose_level - 1);
    cur_sparsity = f->num_nodes;
    if ((cur_sparsity < s_low) && (input_data->verbose_level >= 2)) {
        printf("Warning");
    }
    if (input_data->verbose_level >= 2) {
        printf("Returning final for lambda_l= %d)", cur_sparsity);
    }
    for (i = 0; i < f->num_nodes; i++) {
        proj_w[i] = f->nodes[i];
    }
    return f;
}


/** This solves logistic regression problem with subspace constraint.*/
double *f_min(double *x, double *y, int n, int p, double *w0, int max_iter) {
    if (max_iter <= 0) { // setting maximal number of iterations.
        max_iter = 50;
    }
    double *grad_loss = (double *) malloc(sizeof(double) * (p + 1));
    double *tmp_estimate = (double *) malloc(sizeof(double) * p);
    double *w_hat = (double *) malloc(sizeof(double) * p);
    cblas_dcopy(p, w_hat, 1, w0, 1);
    double *losses = (double *) malloc(sizeof(double) * max_iter);
    double beta, l_rate, nrm_grad, nrm_square;
    for (int i = 0; i < max_iter; i++) {
        batch_logistic_grad(x, y, w_hat, n, p, grad_loss);
        losses[i] = grad_loss[p];
        nrm_grad = cblas_dnrm2(p, grad_loss, 1);
        beta = 0.8, l_rate = 1., nrm_square = 0.5 * nrm_grad * nrm_grad;
        while (true) {
            cblas_dcopy(p, w_hat, 1, tmp_estimate, 1);
            batch_logistic_grad(x, y, w_hat, n, p, grad_loss);
            if (grad_loss[p] > (grad_loss[p] - l_rate * nrm_square)) {
                l_rate = beta * l_rate;
            } else {
                break;
            }
        }
        cblas_daxpy(p, -l_rate, grad_loss, 1, w_hat, 1);
    }
    free(losses);
    free(tmp_estimate);
    free(grad_loss);
    return w_hat;
}

/****************************************************************************
 *
 * @param input_data All input parameters used in online_graph_ghtp
 * @return [wt, run_time, loss]
 ***************************************************************************/
PyObject *run_algo(InputData *input_data) {

    // Generate results
    PyObject *results = PyTuple_New(3);
    PyObject *all_wt = PyList_New(input_data->size);
    PyObject *run_times = PyList_New(input_data->n);
    PyObject *losses = PyList_New(input_data->n);
    // some temp variables
    int p = (int) input_data->p, n = (int) input_data->n;
    double *estimate = (double *) malloc(sizeof(double) * p);
    double *tmp = (double *) malloc(sizeof(double) * p);
    double *wt = (double *) malloc(sizeof(double) * p);
    double *xt = (double *) malloc(sizeof(double) * p);
    double *grad_loss = (double *) malloc(sizeof(double) * (p + 1));
    double loss, yt;
    for (int tt = 1; tt <= n; tt++) {
        int index = tt - 1;
        cblas_dcopy(p, input_data->x + index * p, 1, xt, 1); // receive xt
        yt = input_data->y[index];
        logistic_grad(p, xt, yt, wt, -1, grad_loss);
        cblas_dcopy(p, grad_loss, 1, estimate, 1);
        loss = grad_loss[p];
        free(grad_loss);
        cblas_daxpy(p, -1. / sqrt(tt * 1.), estimate, 1, wt, 1);
        Forest *f = binary_search_tail(wt, input_data);
        cblas_dcopy(p, f->proj_w, 1, tmp, 1);
        // get sub-matrix x,y, TODO, this part need to test.
        int sub_size = f->num_nodes * tt;
        double *sub_x = (double *) malloc(sizeof(double) * sub_size);
        double *sub_y = (double *) malloc(sizeof(double) * sub_size);
        double *sub_w0 = (double *) malloc(sizeof(double) * f->num_nodes);
        for (int ind = 0; ind < f->num_nodes; ind++) {
            int node_i = f->nodes[ind];
            cblas_dcopy(tt, input_data->x + node_i, p,
                        sub_x + ind, f->num_nodes);
        }
        cblas_zcopy(tt, input_data->y, 1, sub_y, 1);
        double *w_hat = f_min(sub_x, sub_y, tt, f->num_nodes, sub_w0, 0);
        cblas_dscal(p, 0.0, wt, 1); // clear to zero
        for (int ind = 0; ind < f->num_nodes; ind++) {
            wt[f->nodes[ind]] = w_hat[ind];
        }
        free(f);
        f = binary_search_tail(wt, input_data);
        wt = f->proj_w;
        if (tt % 100 == 0) {
            printf("tt:%3d loss: %.4lf", tt, loss);
        }
        //to save current loss, wt and
        PyList_SetItem(losses, index, PyFloat_FromDouble(loss));
        PyList_SetItem(run_times, index, PyFloat_FromDouble(0.0));
        for (int ind = 0; ind < input_data->p; ind++) {
            PyObject *item = PyFloat_FromDouble(wt[ind]);
            PyList_SetItem(all_wt, index * p + ind, item);
        }
    }
    PyTuple_SetItem(results, 0, all_wt);
    PyTuple_SetItem(results, 1, run_times);
    PyTuple_SetItem(results, 2, losses);
    return results;
}

/*****************************************************************************
 *
 * This function is a Python interface.
 * @param self just a NULL object.
 * @param args
 *          x: x.shape=(n,p) , where n is # of samples. p: # of features.
 *          y: y.shape=(n, ) labels, each element from {+1,-1}
 *          w0: w0.shape=(p, ) an initial point.
 *          edges: edges.shape=(m,2), where m is # of edges in the graph.
 *          edge_costs: edges.shape=(m, ) edge costs.
 *          pathway_id:            string
 *          s_low:                 int, lower bound of sparsity
 *          s_high:                int, upper bound of sparsity
 *          edge_costs_multiplier: int, to leverage the edge costs
 *          root:                  int, root for pcst, -1 (no root).
 *          num_active_cluster:    int, number of forest, default is 1.
 *          pruning                string, ['strong','gw']. default is 'strong'
 *          verbose_level:         int, >=2 print more details.
 *          max_num_iter:          int, maximal number of iterations in
 *                                      binary search of graph projection.
 * @return
 ****************************************************************************/
static PyObject *online_graph_ghtp(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error.");
        exit(0);
    }
    PyArrayObject *input_x, *input_y, *input_w0;
    PyArrayObject *input_edges, *input_edge_costs;
    unsigned int s_low, s_high, edge_costs_multiplier, root;
    unsigned int num_active_cluster, verbose_level, max_num_iter;
    char *pathway_id, *pruning;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!ziiiiizii",
                          &PyArray_Type, &input_x,
                          &PyArray_Type, &input_y,
                          &PyArray_Type, &input_w0,
                          &PyArray_Type, &input_edges,
                          &PyArray_Type, &input_edge_costs,
                          &pathway_id, &s_low, &s_high,
                          &edge_costs_multiplier, &root,
                          &num_active_cluster, &pruning,
                          &verbose_level, &max_num_iter)) {
        printf("something wrong");
        exit(-1);
    }
    // build some parameters
    int dim_x = (*input_x).nd, i, j;
    long int len_x = PyArray_Size((PyObject *) input_x);
    long int len_y = PyArray_Size((PyObject *) input_y);
    long int len_w0 = PyArray_Size((PyObject *) input_w0);
    long int len_edges = PyArray_Size((PyObject *) input_edges);
    long int len_edge_costs = PyArray_Size((PyObject *) input_edge_costs);
    long int n = input_x->dimensions[0], p = input_x->dimensions[1];
    if ((len_y != n) || (p != len_w0) || (len_edges / 2 != len_edge_costs)) {
        printf("input dimension inconsistent!");
        exit(-1);
    }
    // create input data x and labels y
    double *x = (double *) malloc(sizeof(double) * len_x);
    double *y = (double *) malloc(sizeof(double) * len_y);
    for (i = 0; i < input_x->dimensions[0]; i++) {
        printf("%3d\t", i);
        for (j = 0; j < input_x->dimensions[1]; j++) {
            double *element = (double *) PyArray_GETPTR2(input_x, i, j);
            x[i * p + j] = *element;
        }
        double *yi = (double *) PyArray_GETPTR1(input_y, i);
        y[i] = *yi;
        printf("sum(x): %lf y[%3d]: %lf\n", sum(p, x + i * p), i, y[i]);
    }
    // create initial point w0
    double *w0 = (double *) malloc(sizeof(double) * len_w0);
    for (i = 0; i < len_w0; i++) {
        double *element = (double *) PyArray_GETPTR1(input_w0, i);
        w0[i] = *element;
    }
    // create input edges
    Edge *edges = (Edge *) malloc(sizeof(Edge) * (len_edges / 2));
    printf("num_edges: %ld, edge_size: %ld \n",
           input_edges->dimensions[0], input_edges->dimensions[1]);
    for (i = 0; i < input_edges->dimensions[0]; i++) {
        for (j = 0; j < input_edges->dimensions[1]; j++) {
            int *element = (int *) PyArray_GETPTR2(input_edges, i, j);
            edges[i][j] = *element;
        }
    }
    // create edge costs
    double *edge_costs = (double *) malloc(sizeof(double) * len_edge_costs);
    for (i = 0; i < len_edge_costs; i++) {
        double *element = (double *) PyArray_GETPTR1(input_edge_costs, i);
        edge_costs[i] = *element;
    }

    printf("s_low: %u s_high: %u dimension of x: %d\n", s_low, s_high, dim_x);
    printf("n: %ld p: %ld size of x: %ld\n", n, p, (n * p));
    printf("pathway_id: %s\n", pathway_id);
    printf("pruning: %s\n", pruning);

    InputData *input_data = (InputData *) malloc(sizeof(InputData));
    input_data->n = n;
    input_data->p = p;
    input_data->size = n * p;
    input_data->x = x;
    input_data->y = y;
    input_data->w0 = w0;
    input_data->s_low = s_low;
    input_data->s_high = s_high;
    input_data->edges = edges;
    input_data->edge_costs = edge_costs;
    input_data->verbose_level = verbose_level;
    input_data->root = root;
    input_data->pruning = pruning;
    input_data->pathway_id = pathway_id;
    input_data->edge_costs_multiplier = edge_costs_multiplier;
    input_data->num_active_clusters = num_active_cluster;
    input_data->max_num_iter = max_num_iter;

    PyObject *result = run_algo(input_data);
    if (result == NULL) {
        perror("something wrong !");
        exit(0);
    }
    // to free memory
    free(input_data);
    free(edge_costs);
    free(edges);
    free(w0);
    free(y);
    free(x);
    return result;
}

bool test_logistic() {
    printf("%.16f \n", logistic(10));
    printf("%.16f \n", logistic(-10));
    printf("%.16f \n", logistic(50));
    printf("%.16f \n", logistic(-50));
    printf("%.16f \n", logistic(100));
    printf("%.16f \n", logistic(-100));
    printf("%.16f \n", logistic(1e6));
    printf("%.16f \n", logistic(-1e6));
    return true;
}

bool test_product_v() {
    double *x = (double *) malloc(sizeof(double) * 10);
    for (int i = 0; i < 10; i++) {
        x[i] = -i * i * 1.;
        printf("%.2lf ", x[i]);
    }
    printf("\n");
    self_product(10, x);
    for (int i = 0; i < 10; i++) {
        printf("%.2lf ", x[i]);
    }
    printf("\n");
    memset(x, 0, sizeof(double) * 10);
    for (int i = 0; i < 10; i++) {
        printf("%.2lf ", x[i]);
    }
    printf("\n");
    return true;
};

bool test_all() {
    if (test_logistic()) {
        printf("test logistic passed !\n");
    }
    if (test_product_v()) {
        printf("test self product passed !\n");
    }
}

int main() {
    test_all();
}

/*  define functions in module */
static PyMethodDef CosMethods[] = {
        {"online_graph_ghtp", online_graph_ghtp,
                METH_VARARGS, "Online Graph GHTP docs"},
        {NULL, NULL, 0, NULL}};

#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,
        "online_graph_ghtp", "Some documentation",
        -1,
        CosMethods
};

PyMODINIT_FUNC PyInit_cos_module(void) {
    return PyModule_Create(&cModPyDem);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC initonline_graph_ghtp(void) {
    import_array();
    (void) Py_InitModule("online_graph_ghtp", CosMethods);
}

#endif