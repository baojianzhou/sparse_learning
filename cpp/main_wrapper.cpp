/**========================================================================
 * This is a python wrapper for head and tail projection.
 * Created by Baojian Zhou, Email: bzhou6@albany.edu
 * Date: 06/15/2018
 * License: MIT License
 * =======================================================================*/
#include <cstdio>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "head_tail.h"


// edges, weights, x, g, s, root, max_iter, budget, delta
static PyObject *proj_head(PyObject *self, PyObject *args) {
    if (self != nullptr) {
        printf("unknown error.");
        return nullptr;
    }
    PyArrayObject *edges_, *edge_weights_, *vector_x_;
    unsigned int g, s, root, max_iter;
    double budget, delta;
    if (!PyArg_ParseTuple(args, "O!O!O!iiiidd",
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &edge_weights_,
                          &PyArray_Type, &vector_x_,
                          &g, &s, &root, &max_iter, &budget, &delta)) {
        return nullptr;
    }
    long n = PyArray_Size((PyObject *) vector_x_); // number of nodes
    long m = PyArray_Size((PyObject *) edges_); // number of edges
    cout << "number of nodes: " << n << " number of edges: " << m << endl;
    vector<pair<int, int> > edges; //1. edges are the same.
    vector<double> costs; //3. cost c(e) = w(e) + B / s
    vector<double> prizes; //2. prizes.
    for (size_t i = 0; i < edges_->dimensions[0]; i++) {
        pair<int, int> edge = std::make_pair(-1, -1);
        for (size_t j = 0; j < edges_->dimensions[1]; j++) {
            auto *element = (int *) PyArray_GETPTR2(edges_, i, j);
            if (j == 0) {
                edge.first = *element;
            } else {
                edge.second = *element;
            }
        }
        edges.push_back(edge);
        auto *wei = (double *) PyArray_GETPTR1(edge_weights_, i);
        costs.push_back(*wei + budget / (s * 1.0));
    }
    for (size_t i = 0; i < n; i++) {
        auto *element = (double *) PyArray_GETPTR1(vector_x_, i);
        prizes.push_back((*element) * (*element));
    }
    double C = 2. * budget; //4. cost budget C
    HeadApprox head(edges, costs, prizes, g, s, C, delta);
    pair<vector<int>, vector<int>> f = head.run();
    //5. package the result nodes and edges.
    PyObject *results = PyTuple_New(3);
    PyObject *re_nodes = PyList_New(f.first.size());
    PyObject *re_edges = PyList_New(f.second.size());
    PyObject *p_x = PyList_New(n);
    for (size_t i = 0; i < n; i++) {
        PyList_SetItem(p_x, i, PyFloat_FromDouble(0.0));
    }
    for (size_t i = 0; i < f.first.size(); i++) {
        PyList_SetItem(re_nodes, i, PyInt_FromLong(f.first[i]));
        auto *xi = (double *) PyArray_GETPTR1(vector_x_, f.first[i]);
        PyList_SetItem(p_x, f.first[i], PyFloat_FromDouble(*xi));
    }
    for (size_t i = 0; i < f.second.size(); i++) {
        PyList_SetItem(re_edges, i, PyInt_FromLong(f.second[i]));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    PyTuple_SetItem(results, 1, re_edges);
    PyTuple_SetItem(results, 2, p_x);
    return results;
}

// edges, weights, x, g, s, root, max_iter, budget, nu
static PyObject *proj_tail(PyObject *self, PyObject *args) {
    if (self != nullptr) {
        printf("unknown error.");
        exit(0);
    }
    PyArrayObject *edges_, *edge_weights_, *vector_x_;
    unsigned int g, s, root, max_iter;
    double budget, nu;
    if (!PyArg_ParseTuple(args, "O!O!O!iiiidd",
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &edge_weights_,
                          &PyArray_Type, &vector_x_,
                          &g, &s, &root, &max_iter, &budget, &nu)) {
        return nullptr;
    }
    long int n = PyArray_Size((PyObject *) vector_x_); // number of nodes
    long int m = PyArray_Size((PyObject *) edges_); // number of edges


    //1. edges are the same.
    //2. prizes.
    //3. cost c(e) = w(e) + B / s.
    //2. prizes.
    vector<pair<int, int> > edges;
    vector<double> costs;
    vector<double> prizes;
    for (size_t i = 0; i < edges_->dimensions[0]; i++) {
        pair<int, int> edge = std::make_pair(-1, -1);
        for (size_t j = 0; j < edges_->dimensions[1]; j++) {
            auto *element = (int *) PyArray_GETPTR2(edges_, i, j);
            if (j == 0) {
                edge.first = *element;
            } else {
                edge.second = *element;
            }
        }
        edges.push_back(edge);
        auto *wei = (double *) PyArray_GETPTR1(edge_weights_, i);
        costs.push_back(*wei + budget / (s * 1.0));
    }
    for (size_t i = 0; i < n; i++) {
        auto *element = (double *) PyArray_GETPTR1(vector_x_, i);
        prizes.push_back((*element) * (*element));
    }
    //4. cost budget C.
    double C = 2. * budget;
    //5. delta = min(0.5,1/nu)
    double delta = min(0.5, 1. / nu);
    TailApprox tail(edges, costs, prizes, g, s, C, nu, delta);
    pair<vector<int>, vector<int>> f = tail.run();
    //5. package the result nodes and edges.
    PyObject *results = PyTuple_New(3);
    PyObject *re_nodes = PyList_New(f.first.size());
    PyObject *re_edges = PyList_New(f.second.size());
    PyObject *p_x = PyList_New(n);
    for (size_t i = 0; i < n; i++) {
        PyList_SetItem(p_x, i, PyFloat_FromDouble(0.0));
    }
    for (size_t i = 0; i < f.first.size(); i++) {
        PyList_SetItem(re_nodes, i, PyInt_FromLong(f.first[i]));
        auto *xi = (double *) PyArray_GETPTR1(vector_x_, f.first[i]);
        PyList_SetItem(p_x, f.first[i], PyFloat_FromDouble(*xi));
    }
    for (size_t i = 0; i < f.second.size(); i++) {
        PyList_SetItem(re_edges, i, PyInt_FromLong(f.second[i]));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    PyTuple_SetItem(results, 1, re_edges);
    PyTuple_SetItem(results, 2, p_x);
    return results;
}


//edges, prizes, costs, root, num_clusters, pruning, verbosity_level
static PyObject *proj_pcst(PyObject *self, PyObject *args) {
    if (self != nullptr) {
        printf("unknown error.");
        exit(0);
    }
    PyArrayObject *edges_, *prizes_, *edge_weights_;
    unsigned int g, root, verbose_level;
    char *pruning;
    if (!PyArg_ParseTuple(args, "O!O!O!iizi",
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &prizes_,
                          &PyArray_Type, &edge_weights_,
                          &root, &g, &pruning, &verbose_level)) {
        return nullptr;
    }
    long int n = PyArray_Size((PyObject *) prizes_); // number of nodes
    long int m = PyArray_Size((PyObject *) edges_); // number of edges
    vector<pair<int, int> > edges;
    vector<double> costs;
    vector<double> prizes;
    for (size_t i = 0; i < edges_->dimensions[0]; i++) {
        pair<int, int> edge = std::make_pair(-1, -1);
        for (size_t j = 0; j < edges_->dimensions[1]; j++) {
            auto *element = (int *) PyArray_GETPTR2(edges_, i, j);
            if (j == 0) {
                edge.first = *element;
            } else {
                edge.second = *element;
            }
        }
        edges.push_back(edge);
        auto *wei = (double *) PyArray_GETPTR1(edge_weights_, i);
        costs.push_back(*wei);
    }
    for (size_t i = 0; i < n; i++) {
        auto *element = (double *) PyArray_GETPTR1(prizes_, i);
        prizes.push_back((*element));
    }
    cout << pruning << endl;
    PCSTFast::PruningMethod pruning_method = PCSTFast::parse_pruning_method(
            pruning);
    cout << pruning_method << endl;
    PCSTFast pcst_pcst(edges, prizes, costs, root, g,
                       pruning_method, verbose_level, nullptr);
    vector<int> result_nodes;
    vector<int> result_edges;
    cout << pruning_method << endl;
    cout << "test 1" << endl;
    if (!pcst_pcst.run(&result_nodes, &result_edges)) {
        cout << "bad thing happened" << endl;
    }
    cout << pruning_method << endl;
    cout << "test 2" << endl;
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


/**
 * define functions in the proj module.
 * 1. function name in your Python program,
 * 2. function name defined in c program,
 * 3. flags, usually is METH_VARARGS
 * 4. some docs.
 *
 * Here we defined 3 functions.
 */
static PyMethodDef proj_methods[] = {
        {"proj_head", (PyCFunction) proj_head, METH_VARARGS, "Head docs"},
        {"proj_tail", (PyCFunction) proj_tail, METH_VARARGS, "Tail docs"},
        {"proj_pcst", (PyCFunction) proj_pcst, METH_VARARGS, "PCST docs"},
        {nullptr, nullptr, 0, nullptr}};

#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,"proj_algo", "Some documentation",
        -1, proj_methods};

PyMODINIT_FUNC PyInit_cos_module(void) {
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
    cout << "test" << endl;
    return 0;
}
