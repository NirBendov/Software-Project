#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "symnmf.h"

//converts python 2d list into C matrix
static double** get_matrix(PyObject* arr, int* row_count, int* row_len) {
    int i, j, row_num, col_num;
    double num, ** matrix;
    PyObject* vect, * item;

    row_num = PyList_Size(arr); // number of rows in matrix
    matrix = (double**)calloc(row_num, sizeof(double*));
    if (matrix == NULL) {
        free(matrix);
        return NULL;
    }
    for (i = 0; i < row_num; ++i) {
        vect = PyList_GetItem(arr, i); // vect = arr[i]
        col_num = PyList_Size(vect); // number of columns in matrix
        matrix[i] = (double*)calloc(col_num, sizeof(double));
        if (matrix[i] == NULL) {
            free(matrix);
            return NULL;
        }
        for (j = 0; j < col_num; ++j) {
            item = PyList_GetItem(vect, j); // item = vect[j] = arr[i][j]
            num = PyFloat_AsDouble(item);
            matrix[i][j] = num;
        }
    }
    *row_count = row_num;
    *row_len = col_num;
    return matrix;
}

//converts C matrix into python 2d list
static PyObject* py_matrix(double** matrix, int row_count, int row_len) {
    PyObject* result = PyList_New(row_count);
    for (Py_ssize_t i = 0; i < row_count; i++) {
        PyObject* item = PyList_New(row_len);
        for (Py_ssize_t j = 0; j < row_len; j++)
            PyList_SET_ITEM(item, j, PyFloat_FromDouble(matrix[i][j]));
        PyList_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* sym(PyObject* self, PyObject* args) {
    double** X, ** A;
    int row_count, row_len;
    PyObject* mat, * result;
    if (!PyArg_ParseTuple(args, "O", &mat)) {
        return NULL;
    }
    X = get_matrix(mat, &row_count, &row_len);
    if (X == NULL)
        return Py_BuildValue("s", "an error has occured!");
    A = sym_impl(X, row_count, row_len);
    result = py_matrix(A, row_count, row_count);
    free_matrix(A, row_count);
    free_matrix(X, row_count);
    return Py_BuildValue("O", result);
}

static PyObject* ddg(PyObject* self, PyObject* args) {
    double** X, ** D;
    int row_count, row_len;
    PyObject* mat, * result;
    if (!PyArg_ParseTuple(args, "O", &mat)) {
        return NULL;
    }
    X = get_matrix(mat, &row_count, &row_len);
    if (X == NULL)
        return Py_BuildValue("s", "an error has occured!");
    D = ddg_impl(X, row_count, row_len);
    result = py_matrix(D, row_count, row_count);
    free_matrix(D, row_count);
    free_matrix(X, row_count);
    return Py_BuildValue("O", result);
}

static PyObject* norm(PyObject* self, PyObject* args) {
    double** X, ** W;
    int row_count, row_len;
    PyObject* mat, * result;
    if (!PyArg_ParseTuple(args, "O", &mat)) {
        return NULL;
    }
    X = get_matrix(mat, &row_count, &row_len);
    if (X == NULL)
        return Py_BuildValue("s", "an error has occured!");
    W = norm_impl(X, row_count, row_len);
    result = py_matrix(W, row_count, row_count);
    free_matrix(W, row_count);
    free_matrix(X, row_count);
    return Py_BuildValue("O", result);
}

static PyObject* symnmf(PyObject* self, PyObject* args) {
    double** H, ** W;
    int row_count, row_len;
    PyObject* H_py, * W_py, * result;
    if (!PyArg_ParseTuple(args, "OO", &H_py, &W_py)) {
        return NULL;
    }
    W = get_matrix(W_py, &row_count, &row_len);
    H = get_matrix(H_py, &row_count, &row_len);
    if (W == NULL || H == NULL)
        return Py_BuildValue("s", "an error has occured!");
    optimize_H(H, W, row_count, row_len);
    result = py_matrix(H, row_count, row_len);
    free_matrix(H, row_count);
    free_matrix(W, row_count);
    return Py_BuildValue("O", result);
}

static PyMethodDef symMethods[] = {
    {"sym",
      sym,
      METH_VARARGS,
      PyDoc_STR("Calculate and output the similarity matrix")},
    {"ddg",
      ddg,
      METH_VARARGS,
      PyDoc_STR("Calculate and output the Diagonal Degree Matrix")},
    {"norm",
      norm,
      METH_VARARGS,
      PyDoc_STR("Calculate and output the normalized similarity matrix")},
    {"symnmf",
      symnmf,
      METH_VARARGS,
      PyDoc_STR("Perform the full symNMF algorithm")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    symMethods
};

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    PyObject* m;
    m = PyModule_Create(&symnmfmodule);
    if (!m) {
        return NULL;
    }
    return m;
}