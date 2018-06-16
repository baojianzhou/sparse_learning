//
// Created by baojian on 6/16/18.
//

#include <cstdio>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "head_tail.h"


int main() {
    cout << "test" << endl;
    return 0;
}

// edges, weights, x, g, s, root, max_iter, budget, delta
static PyObject *proj_head(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error.");
        exit(0);
    }
    PyArrayObject *edges, *edge_weights, *vector_x;
    unsigned int g, s, root, max_iter;
    double budget, delta;
    if (!PyArg_ParseTuple(args, "O!O!O!iiiidd",
                          &PyArray_Type, &edges,
                          &PyArray_Type, &edge_weights,
                          &PyArray_Type, &vector_x,
                          &g, &s, &root, &max_iter, &budget, &delta)) {
        return NULL;
    }
    long int n = PyArray_Size((PyObject *) vector_x); // number of nodes
    long int m = PyArray_Size((PyObject *) edges); // number of edges
    PyObject *results = PyTuple_New(3);
    PyObject *re_nodes = PyList_New(5);
    PyObject *re_edges = PyList_New(4);
    PyObject *p_x = PyList_New(n);

    PyList_SetItem(re_nodes, 0, PyInt_FromLong(10));
    PyList_SetItem(re_nodes, 1, PyInt_FromLong(11));
    PyList_SetItem(re_nodes, 2, PyInt_FromLong(12));
    PyList_SetItem(re_nodes, 3, PyInt_FromLong(13));
    PyList_SetItem(re_nodes, 4, PyInt_FromLong(14));
    PyList_SetItem(re_edges, 0, PyInt_FromLong(1));
    PyList_SetItem(re_edges, 1, PyInt_FromLong(2));
    PyList_SetItem(re_edges, 2, PyInt_FromLong(3));
    PyList_SetItem(re_edges, 3, PyInt_FromLong(5));
    for (int i = 0; i < n; i++) {
        PyList_SetItem(p_x, i, PyFloat_FromDouble(i + 0.5));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    PyTuple_SetItem(results, 1, re_edges);
    PyTuple_SetItem(results, 2, p_x);
    return results;
}

// edges, weights, x, g, s, root, max_iter, budget, nu
static PyObject *proj_tail(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error.");
        exit(0);
    }
    PyArrayObject *edges, *edge_weights, *vector_x;
    unsigned int g, s, root, max_iter;
    double budget, nu;
    if (!PyArg_ParseTuple(args, "O!O!O!iiiidd",
                          &PyArray_Type, &edges,
                          &PyArray_Type, &edge_weights,
                          &PyArray_Type, &vector_x,
                          &g, &s, &root, &max_iter, &budget, &nu)) {
        return NULL;
    }
    long int n = PyArray_Size((PyObject *) vector_x); // number of nodes
    long int m = PyArray_Size((PyObject *) edges); // number of edges
    PyObject *results = PyTuple_New(3);
    PyObject *re_nodes = PyList_New(5);
    PyObject *re_edges = PyList_New(4);
    PyObject *p_x = PyList_New(n);

    PyList_SetItem(re_nodes, 0, PyInt_FromLong(10));
    PyList_SetItem(re_nodes, 1, PyInt_FromLong(11));
    PyList_SetItem(re_nodes, 2, PyInt_FromLong(12));
    PyList_SetItem(re_nodes, 3, PyInt_FromLong(13));
    PyList_SetItem(re_nodes, 4, PyInt_FromLong(14));
    PyList_SetItem(re_edges, 0, PyInt_FromLong(1));
    PyList_SetItem(re_edges, 1, PyInt_FromLong(2));
    PyList_SetItem(re_edges, 2, PyInt_FromLong(3));
    PyList_SetItem(re_edges, 3, PyInt_FromLong(5));
    for (int i = 0; i < n; i++) {
        PyList_SetItem(p_x, i, PyFloat_FromDouble(i + 0.5));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    PyTuple_SetItem(results, 1, re_edges);
    PyTuple_SetItem(results, 2, p_x);
    return results;
}

//edges, weights, prizes, root, g, pruning, verbose_level
static PyObject *proj_pcst(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error.");
        exit(0);
    }
    PyArrayObject *edges, *edge_weights, *prizes;
    unsigned int g, root, verbose_level;
    char *pruning;
    if (!PyArg_ParseTuple(args, "O!O!O!iizi",
                          &PyArray_Type, &edges,
                          &PyArray_Type, &edge_weights,
                          &PyArray_Type, &prizes,
                          &g, &root, &pruning, &verbose_level)) {
        return NULL;
    }
    long int n = PyArray_Size((PyObject *) prizes); // number of nodes
    long int m = PyArray_Size((PyObject *) edges); // number of edges
    PyObject *results = PyTuple_New(2);
    PyObject *re_nodes = PyList_New(5);
    PyObject *re_edges = PyList_New(4);

    PyList_SetItem(re_nodes, 0, PyInt_FromLong(10));
    PyList_SetItem(re_nodes, 1, PyInt_FromLong(11));
    PyList_SetItem(re_nodes, 2, PyInt_FromLong(12));
    PyList_SetItem(re_nodes, 3, PyInt_FromLong(13));
    PyList_SetItem(re_nodes, 4, PyInt_FromLong(14));
    PyList_SetItem(re_edges, 0, PyInt_FromLong(1));
    PyList_SetItem(re_edges, 1, PyInt_FromLong(2));
    PyList_SetItem(re_edges, 2, PyInt_FromLong(3));
    PyList_SetItem(re_edges, 3, PyInt_FromLong(5));
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
        {NULL, NULL, 0, NULL}};

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