#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>

typedef float f32;
typedef double f64;

int main() {
    printf("Hello, World!\n");
    return 0;
}

static PyObject *proj_head(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error.");
        exit(0);
    }
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyObject *proj_tail(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error.");
        exit(0);
    }
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyObject *proj_pcst(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error.");
        exit(0);
    }
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
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
static PyMethodDef ProjMethods[] = {
        {"proj_head", (PyCFunction) proj_head, METH_VARARGS, "Head docs"},
        {"proj_tail", (PyCFunction) proj_tail, METH_VARARGS, "Tail docs"},
        {"proj_pcst", (PyCFunction) proj_pcst, METH_VARARGS, "PCST docs"},
        {NULL, NULL, 0, NULL}};

#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem = {
        PyModuleDef_HEAD_INIT,"head_proj", "Some documentation",
        -1, ProjMethods};

PyMODINIT_FUNC PyInit_cos_module(void) {
    return PyModule_Create(&cModPyDem);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC initProjModule() {
    Py_InitModule3("proj_head", ProjMethods, "some docs for head proj.");
    Py_InitModule3("proj_tail", ProjMethods, "some docs for head proj.");
    Py_InitModule3("proj_pcst", ProjMethods, "some docs for head proj.");
}

#endif


