//
// Created by baojian on 7/1/18.
//
#include <Python.h>
#include <stdio.h>

static PyObject *method_1(PyObject *self, PyObject *args) {
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}


/**
 * Here we defined 4 functions.
 *
 * 1. proj_head
 * 2. proj_tail
 * 3. proj_pcst
 * 4. minimal_spanning_tree
 *
 * above 3 functions had been tested on Python2.7.
 *
 * defined functions in the proj module.
 * 1. function name in your Python program,
 * 2. function name defined in c program,
 * 3. flags, usually is METH_VARARGS
 * 4. some docs.
 */
static PyMethodDef all_methods_table[] = {
        {"method_1", (PyCFunction) method_1, METH_VARARGS, "method 1 docs"},
        {NULL, NULL, 0, NULL}};

/** Only support for Python version 3*/
static struct PyModuleDef sparse_learning_module = {
        PyModuleDef_HEAD_INIT, "sparse_learning_module", "Some documentation",
        -1, all_methods_table};

PyMODINIT_FUNC PyInit_sparse_learning_module(void) {
    return PyModule_Create(&sparse_learning_module);
}


int main() {
    printf("method_1");
}