/**========================================================================
 * This is a python wrapper for head and tail projection.
 * Created by Baojian Zhou, Email: bzhou6@albany.edu
 * Date: 06/15/2018
 * License: MIT License
 * =======================================================================*/
#include <cstdio>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "pcst_fast.h"
#include "head_tail.h"
#include "union_find.h"
#include "ghtp_algo.h"

using std::cerr;
using cluster_approx::PCSTFast;

static PyObject *proj_head(PyObject *self, PyObject *args) {
    /**
     * DO NOT call this function directly, use the Python Wrapper instead.
     * list of args:
     * args[0]: ndarray dim=(m,2) -- edges of the graph.
     * args[1]: ndarray dim=(m,)  -- weights (positive) of the graph.
     * args[2]: ndarray dim=(n,)  -- the vector needs to be projected.
     * args[3]: integer np.int32  -- number of connected components returned.
     * args[4]: integer np.int32  -- sparsity (positive) parameter.
     * args[5]: double np.float64 -- budget of the graph model.
     * args[6]: double np.float64 -- delta. default is 1. / 169.
     * args[7]: integer np.int32  -- maximal # of iterations in the loop.
     * args[8]: double np.float64 -- error tolerance for minimum nonzero.
     * args[9]: integer np.int32  -- root(default is -1).
     * args[10]: string string    -- pruning ['simple', 'gw', 'strong'].
     * args[11]: double np.float64-- epsilon to control the presion of PCST.
     * args[12]: integer np.int32 -- verbosity level
     * @return: (re_nodes, re_edges, p_x)
     * re_nodes: projected nodes
     * re_edges: projected edges (indices)
     * p_x: projection of x.
     */
    if (self != nullptr) {
        cerr << "unknown error for no reason." << endl;
        return nullptr;
    }
    PyArrayObject *edges_, *edge_weights_, *vector_x_;
    int g, s, root, max_iter, verbose;
    double budget, delta, epsilon, err_tol;
    char *pruning;
    if (!PyArg_ParseTuple(
            args, "O!O!O!iiddidizdi", &PyArray_Type, &edges_, &PyArray_Type,
            &edge_weights_, &PyArray_Type, &vector_x_, &g, &s, &budget, &delta,
            &max_iter, &err_tol, &root, &pruning, &epsilon, &verbose)) {
        return nullptr;
    }
    long n = vector_x_->dimensions[0];  // number of nodes
    long m = edges_->dimensions[0];     // number of edges
    vector<pair<int, int> > edges;
    vector<double> costs;
    vector<double> prizes;
    vector<double> vector_x;
    PyObject *results = PyTuple_New(3);
    PyObject *p_x = PyList_New(n);      // projected x
    for (size_t i = 0; i < m; i++) {
        auto *u = (int *) PyArray_GETPTR2(edges_, i, 0);
        auto *v = (int *) PyArray_GETPTR2(edges_, i, 1);
        pair<int, int> edge = std::make_pair(*u, *v);
        edges.push_back(edge);
        auto *wei = (double *) PyArray_GETPTR1(edge_weights_, i);
        costs.push_back(*wei + budget / s);
    }
    for (size_t i = 0; i < n; i++) {
        auto *xi = (double *) PyArray_GETPTR1(vector_x_, i);
        vector_x.push_back(*xi);
        prizes.push_back((*xi) * (*xi));
        PyList_SetItem(p_x, i, PyFloat_FromDouble(0.0));
    }
    double C = 2. * budget;
    HeadApprox head(edges, costs, prizes, g, s, C, delta, max_iter, err_tol,
                    root, pruning, epsilon, verbose);
    pair<vector<int>, vector<int>> f = head.run();
    PyObject *re_nodes = PyList_New(f.first.size());
    PyObject *re_edges = PyList_New(f.second.size());
    for (size_t i = 0; i < f.first.size(); i++) {
        auto node_i = f.first[i];
        PyList_SetItem(re_nodes, i, PyInt_FromLong(node_i));
        PyList_SetItem(p_x, node_i, PyFloat_FromDouble(vector_x[node_i]));
    }
    for (size_t i = 0; i < f.second.size(); i++) {
        PyList_SetItem(re_edges, i, PyInt_FromLong(f.second[i]));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    PyTuple_SetItem(results, 1, re_edges);
    PyTuple_SetItem(results, 2, p_x);
    return results;
}

static PyObject *proj_tail(PyObject *self, PyObject *args) {
    /**
     * DO NOT call this function directly, use the Python Wrapper instead.
     * list of args:
     * args[0]: ndarray dim=(m,2) -- edges of the graph.
     * args[1]: ndarray dim=(m,)  -- weights (positive) of the graph.
     * args[2]: ndarray dim=(n,)  -- the vector needs to be projected.
     * args[3]: integer np.int32  -- number of connected components returned.
     * args[4]: integer np.int32  -- sparsity (positive) parameter.
     * args[5]: double np.float64 -- budget of the graph model.
     * args[6]: double np.float64 -- nu. default is 2.5
     * args[7]: integer np.int32  -- maximal # of iterations in the loop.
     * args[8]: double np.float32 -- error tolerance for minimum nonzero.
     * args[9]: integer np.int32  -- root(default is -1).
     * args[10]: string string    -- pruning ['simple', 'gw', 'strong'].
     * args[11]: double np.float64-- epsilon to control the presion of PCST.
     * args[12]: integer np.int32 -- verbosity level
     * @return: (re_nodes, re_edges, p_x)
     * re_nodes: projected nodes
     * re_edges: projected edges (indices)
     * p_x: projection of x.
     */
    if (self != nullptr) {
        cerr << "unknown error for no reason." << endl;
        return nullptr;
    }
    PyArrayObject *edges_, *edge_weights_, *vector_x_;
    int g, s, root, max_iter, verbose;
    double budget, nu, epsilon, err_tol;
    char *pruning;
    if (!PyArg_ParseTuple(
            args, "O!O!O!iiddidizdi", &PyArray_Type, &edges_, &PyArray_Type,
            &edge_weights_, &PyArray_Type, &vector_x_, &g, &s, &budget, &nu,
            &max_iter, &err_tol, &root, &pruning, &epsilon, &verbose)) {
        return nullptr;
    }
    long n = vector_x_->dimensions[0];  // number of nodes
    long m = edges_->dimensions[0];     // number of edges
    vector<pair<int, int> > edges;
    vector<double> costs;
    vector<double> prizes;
    vector<double> vector_x;
    PyObject *results = PyTuple_New(3);
    PyObject *p_x = PyList_New(n);      // projected x
    for (size_t i = 0; i < m; i++) {
        auto *u = (int *) PyArray_GETPTR2(edges_, i, 0);
        auto *v = (int *) PyArray_GETPTR2(edges_, i, 1);
        pair<int, int> edge = std::make_pair(*u, *v);
        edges.push_back(edge);
        auto *wei = (double *) PyArray_GETPTR1(edge_weights_, i);
        costs.push_back(*wei + budget / s);
    }
    for (size_t i = 0; i < n; i++) {
        auto *xi = (double *) PyArray_GETPTR1(vector_x_, i);
        vector_x.push_back(*xi);
        prizes.push_back((*xi) * (*xi));
        PyList_SetItem(p_x, i, PyFloat_FromDouble(0.0));
    }
    double C = 2. * budget;
    TailApprox tail(edges, costs, prizes, g, s, C, nu, max_iter, err_tol,
                    root, pruning, epsilon, verbose);
    pair<vector<int>, vector<int>> f = tail.run();
    PyObject *re_nodes = PyList_New(f.first.size());
    PyObject *re_edges = PyList_New(f.second.size());
    for (size_t i = 0; i < f.first.size(); i++) {
        auto node_i = f.first[i];
        PyList_SetItem(re_nodes, i, PyInt_FromLong(node_i));
        PyList_SetItem(p_x, node_i, PyFloat_FromDouble(vector_x[node_i]));
    }
    for (size_t i = 0; i < f.second.size(); i++) {
        PyList_SetItem(re_edges, i, PyInt_FromLong(f.second[i]));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    PyTuple_SetItem(results, 1, re_edges);
    PyTuple_SetItem(results, 2, p_x);
    return results;
}

static PyObject *proj_pcst(PyObject *self, PyObject *args) {
    /**
     * DO NOT call this function directly, use the Python Wrapper instead.
     * list of args:
     * args[0]: ndarray dim=(m,2) -- edges of the graph.
     * args[1]: ndarray dim=(n,)  -- prizes of the graph.
     * args[2]: ndarray dim=(m,)  -- costs on nodes.
     * args[3]: integer np.int32  -- root(default is -1).
     * args[4]: integer np.int32  -- number of connected components returned.
     * args[5]: string string     -- pruning none, simple, gw, strong.
     * args[6]: double np.float32 -- epsilon to control the precision.
     * args[7]: integer np.int32  -- verbosity level
     * @return: (re_nodes, re_edges)
     * re_nodes: result nodes
     * re_edges: result edges
     */
    if (self != nullptr) {
        cerr << "unknown error for no reason." << endl;
        return nullptr;
    }
    PyArrayObject *edges_, *prizes_, *edge_weights_;
    int g, root, verbose;
    char *pruning;
    double epsilon;
    if (!PyArg_ParseTuple(args, "O!O!O!iizdi", &PyArray_Type, &edges_,
                          &PyArray_Type, &prizes_, &PyArray_Type,
                          &edge_weights_, &root, &g, &pruning,
                          &epsilon, &verbose)) { return nullptr; }
    long n = prizes_->dimensions[0];    // number of nodes
    long m = edges_->dimensions[0];     // number of edges
    vector<pair<int, int> > edges;
    vector<double> costs;
    vector<double> prizes;
    for (size_t i = 0; i < m; i++) {
        auto *u = (int *) PyArray_GETPTR2(edges_, i, 0);
        auto *v = (int *) PyArray_GETPTR2(edges_, i, 1);
        pair<int, int> edge = std::make_pair(*u, *v);
        edges.push_back(edge);
        auto *wei = (double *) PyArray_GETPTR1(edge_weights_, i);
        costs.push_back(*wei);
    }
    for (size_t i = 0; i < n; i++) {
        auto *element = (double *) PyArray_GETPTR1(prizes_, i);
        prizes.push_back((*element));
    }
    PCSTFast pcst_pcst(edges, prizes, costs, root, g,
                       PCSTFast::parse_pruning_method(pruning),
                       epsilon, verbose, nullptr);
    vector<int> result_nodes;
    vector<int> result_edges;
    if (!pcst_pcst.run(&result_nodes, &result_edges)) {
        cout << "bye bye!" << endl;
    }
    PyObject *results = PyTuple_New(2);
    PyObject *re_nodes = PyList_New(result_nodes.size());
    PyObject *re_edges = PyList_New(result_edges.size());
    for (size_t i = 0; i < result_nodes.size(); i++) {
        PyList_SetItem(re_nodes, i, PyInt_FromLong(result_nodes[i]));
    }
    for (size_t i = 0; i < result_edges.size(); i++) {
        PyList_SetItem(re_edges, i, PyInt_FromLong(result_edges[i]));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    PyTuple_SetItem(results, 1, re_edges);
    return results;
}

static PyObject *mst(PyObject *self, PyObject *args) {
    /**
     * minimal spanning tree
     * DO NOT call this function directly, use the Python Wrapper instead.
     * list of args:
     * args[0]: ndarray dim=(m,2) -- edges of the graph.
     * args[1]: ndarray dim=(m,)  -- weights of the graph.
     * args[2]: integer np.int32  -- number of nodes in the graph.
     * @return: (the edge indices of the spanning tree)
     * re_nodes: result nodes
     * re_edges: result edges
     */
    if (self != nullptr) {
        cerr << "unknown error for no reason." << endl;
        return nullptr;
    }
    PyArrayObject *edges_, *edge_weights_;
    int num_nodes;
    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &edges_,
                          &PyArray_Type, &edge_weights_,
                          &num_nodes)) { return nullptr; }
    long m = edges_->dimensions[0];     // number of edges
    vector<pair<int, int> > edges;
    vector<double> weights;
    for (size_t i = 0; i < m; i++) {
        auto *u = (int *) PyArray_GETPTR2(edges_, i, 0);
        auto *v = (int *) PyArray_GETPTR2(edges_, i, 1);
        pair<int, int> edge = std::make_pair(*u, *v);
        edges.push_back(edge);
        auto *wei = (double *) PyArray_GETPTR1(edge_weights_, i);
        weights.push_back(*wei);
    }
    vector<size_t> selected_edges;
    selected_edges = kruskal_mst(edges, weights, (size_t) (num_nodes));
    PyObject *results = PyList_New(selected_edges.size());
    for (size_t i = 0; i < selected_edges.size(); i++) {
        PyList_SetItem(results, i, PyInt_FromLong(selected_edges[i]));
    }
    return results;
}

static PyObject *ghtp_logistic(PyObject *self, PyObject *args) {
    /**
     * Gradient Hard Thresholding
     * DO NOT call this function directly, use the Python Wrapper instead.
     * list of args:
     * args[0]: ndarray dim=(n,p) -- training data, x_tr
     * args[1]: ndarray dim=(n,)  -- labels, y_tr {+1,-1}
     * args[2]: ndarray dim=(p+1,)-- initial point (including intercept)
     * args[3]: ndarray np.float64-- learning rate
     * args[4]: integer np.int32  -- sparsity parameter
     * args[5]: double  np.float64-- tol tolerance for stop condition
     * args[6]: integer np.int32  -- maximal_iter maximal iterations.
     * args[7]: double  np.int32  -- regularization parameter
     * @return: (wt,losses)
     */
    if (self != nullptr) {
        cerr << "unknown error for no reason." << endl;
        return nullptr;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, sparsity, max_iter, i, j;
    double eta, tol, lr;
    if (!PyArg_ParseTuple(args, "O!O!O!didid",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_,
                          &lr, &sparsity, &tol, &max_iter, &eta)) {
        return nullptr;
    }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    printf("n:%d p:%d\n", n, p);
    auto *x_tr = (double *) malloc(n * p * sizeof(double));
    auto *y_tr = (double *) malloc(n * sizeof(double));
    auto *wt = (double *) malloc((p + 1) * sizeof(double));
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            auto *val = (double *) PyArray_GETPTR2(x_tr_, i, j);
            x_tr[i * p + j] = *val;
        }
        auto *val = (double *) PyArray_GETPTR1(y_tr_, i);
        y_tr[i] = *val;
    }
    for (i = 0; i < (p + 1); i++) {
        auto *val = (double *) PyArray_GETPTR1(w0_, i);
        wt[i] = *val;
    }

    //////////////// gradient hard thresholding pursuit /////////////////////
    auto *loss_grad = (double *) malloc((p + 2) * sizeof(double));
    auto *wt_tmp = (double *) malloc((p + 1) * sizeof(double));
    auto *set_s = (int *) malloc((sparsity + 1) * sizeof(int));
    auto *losses = (double *) malloc((max_iter) * sizeof(double));
    double norm_grad = 0.0;
    for (int tt = 0; tt < max_iter; tt++) {
        loss_logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, n, p);
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        argsort(wt_tmp, sparsity, p + 1, set_s);
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
        min_f(set_s, x_tr, y_tr, wt_tmp, max_iter, eta, wt, n, p, sparsity);
        losses[tt] = loss_grad[0];
        if (tt >= 1 && (abs(losses[tt] - losses[tt - 1]) < tol)) {
            break; // stop earlier when it almost stops decreasing the loss
        }
        for (i = 0; i < p; i++) {
            norm_grad += loss_grad[i + 1] * loss_grad[i + 1];
        }
        printf("losses[%d]: %lf, grad:[%d]: %lf\n",
               tt, losses[tt], tt, sqrt(norm_grad));
    }
    /////////////////////////////////////////////////////////////////////////
    PyObject *results = PyTuple_New(3);
    PyObject *re_wt = PyList_New(p);
    for (i = 0; i < p; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
    }
    PyObject *re_intercept = PyList_New(1);
    PyList_SetItem(re_intercept, 0, PyFloat_FromDouble(wt[p]));
    PyObject *re_losses = PyList_New(max_iter);
    for (i = 0; i < max_iter; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_intercept);
    PyTuple_SetItem(results, 2, re_losses);
    free(losses);
    free(set_s);
    free(wt);
    free(loss_grad);
    free(wt);
    free(y_tr);
    free(x_tr);
    return results;
}

static PyObject *graph_ghtp_logistic(PyObject *self, PyObject *args) {
    //x_tr, y_tr, w0, lr, sparsity, tol, maximal_iter, eta
    /**
     * Gradient Hard Thresholding
     * DO NOT call this function directly, use the Python Wrapper instead.
     * list of args:
     * args[0]: ndarray dim=(n,p) -- training data, x_tr
     * args[1]: ndarray dim=(n,)  -- labels, y_tr {+1,-1}
     * args[2]: ndarray dim=(p+1,)-- initial point (including intercept)
     * args[3]: ndarray np.float64-- learning rate
     * args[4]: integer np.int32  -- sparsity parameter
     * args[5]: double  np.float64-- tol tolerance for stop condition
     * args[6]: integer np.int32  -- maximal_iter maximal iterations.
     * args[7]: double  np.int32  -- regularization parameter
     * @return: (wt,losses)
     */
    if (self != nullptr) {
        cerr << "unknown error for no reason." << endl;
        return nullptr;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, k, max_iter, i, j;
    double eta, tol, lr;
    if (!PyArg_ParseTuple(args, "O!O!O!didid",
                          &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_,
                          &lr, &k, &tol, &max_iter, &eta)) { return nullptr; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    auto *x = (double *) malloc(n * p * sizeof(double));
    auto *y = (double *) malloc(n * sizeof(double));
    auto *wt = (double *) malloc(p * sizeof(double));
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            auto *val = (double *) PyArray_GETPTR2(x_tr_, i, j);
            x[i * p + j] = *val;
            printf("%lf ", x[i * p + j]);
        }
        auto *val = (double *) PyArray_GETPTR1(y_tr_, i);
        y[i] = *val;
        printf("%lf \n", y[i]);
    }
    printf("w0: ");
    for (i = 0; i < (p + 1); i++) {
        auto *val = (double *) PyArray_GETPTR1(w0_, i);
        wt[i] = *val;
        printf(" %lf", wt[i]);
    }


    PyObject *results = PyTuple_New(3);
    PyObject *re_wt = PyList_New(p);
    PyObject *re_intercept = PyList_New(1);
    PyObject *re_losses = PyList_New(max_iter);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_intercept);
    PyTuple_SetItem(results, 2, re_losses);
    free(wt);
    free(y);
    free(x);
    return results;
}


/**
 * Here we defined 6 functions.
 *
 * 1. proj_head
 * 2. proj_tail
 * 3. proj_pcst
 * 4. mst: minimal_spanning_tree
 * 5. ghtp_logistic: gradient hard thresholding pursuit for logistic function.
 * 6. graph_ghtp_logistic: graph-constrained ghtp_logistic
 * above 6 functions had been tested on Python2.7.
 *
 * each function is defined in the proj module.
 * 1. function name in your Python program,
 * 2. function name defined in c program,
 * 3. flags, usually is METH_VARARGS
 * 4. some docs.
 */
static PyMethodDef proj_methods[] = {
        {"proj_head",           (PyCFunction) proj_head, METH_VARARGS, "Head docs"},
        {"proj_tail",           (PyCFunction) proj_tail, METH_VARARGS, "Tail docs"},
        {"proj_pcst",           (PyCFunction) proj_pcst, METH_VARARGS, "PCST docs"},
        {"mst",                 (PyCFunction) mst,       METH_VARARGS, "mst docs"},
        {"ghtp_logistic",       (PyCFunction) ghtp_logistic,
                                                         METH_VARARGS, "ghtp_logistic docs"},
        {"graph_ghtp_logistic", (PyCFunction) graph_ghtp_logistic,
                                                         METH_VARARGS, "graph_ghtp_logistic docs"},
        {nullptr,               nullptr, 0,                            nullptr}};


#if PY_MAJOR_VERSION >= 3

/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,"proj_module", "Some documentation",
        -1, proj_methods};

PyMODINIT_FUNC PyInit_proj_module(void) {
    return PyModule_Create(&cModPyDem);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC initproj_module() {
    Py_InitModule3("proj_module", proj_methods, "some docs for head proj.");
    import_array();
}

#endif

int main() {
    int s = 3;
    int w_len = 5;
    auto *w = (double *) malloc(sizeof(double) * w_len);
    w[0] = -1.;
    w[1] = 1.;
    w[2] = 3.;
    w[3] = -2.;
    w[4] = 3.;
    auto *set_s = (int *) malloc(sizeof(int) * s);
    argsort(w, s, 5, set_s);
}