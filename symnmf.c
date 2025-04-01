#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"
#define eps (0.0001)
#define max_iter (300)
#define beta (0.5)
#define H_ij(h, m1, m2, i, j) ((h[i][j])*(1 - beta + beta*((m1[i][j])/(m2[i][j]))))

void free_matrix(double** matrix, int row_count) {
    int i;
    for (i = 0; i < row_count; ++i) {
        free(matrix[i]);
    }
    free(matrix);
}

/*Computes a_ij for the similarity matrix A, assumes i != j */
double exp_entry(double xi[], double xj[], int l) {
    double d = 0.0; int i;
    for (i = 0; i < l; ++i) {
        d += pow((xi[i] - xj[i]), 2);
    }
    return exp(d*(-0.5));
}

/* Sums up the entries of a given vector */
double sum_vector(double v[], int len) {
    double sum = 0.0; int i;
    for (i = 0; i < len; ++i) {
        sum += v[i];
    }
    return sum;
}

/* Computes the similarity matrix A */
double** sym_impl(double** X, int row_count, int row_len) {
    double **A; int i, j;
    A = (double**)calloc(row_count, sizeof(double*));
    for (i = 0; i < row_count; ++i) {
        A[i] = (double*)calloc(row_count, sizeof(double));
        A[i][i] = 0.0;
        for (j = 0; j < i; ++j) {
            A[j][i] = exp_entry(X[i], X[j], row_len);
            A[i][j] = A[j][i];
        }
    }
    return A;
}

/* Computes the diagonal degree matrix */
double** degree_matrix(double** A, int row_count) {
    double **D; int i;
    D = (double**)calloc(row_count, sizeof(double*));
    for (i = 0; i < row_count; ++i) {
        D[i] = (double*)calloc(row_count, sizeof(double));
        D[i][i] = sum_vector(A[i], row_count);
    }
    return D;
}

double** ddg_impl(double** X, int row_count, int row_len) {
    double **A, **D;
    A = sym_impl(X, row_count, row_len);
    D = degree_matrix(A, row_count);
    free_matrix(A, row_count);
    return D;
}

/* Computes D^(-1/2) */
double** degree_matrix_minus_half_pow(double** D, int len) {
    double **D_pow; int i;
    D_pow = (double**)calloc(len, sizeof(double*));
    for (i = 0; i < len; ++i) {
        D_pow[i] = (double*)calloc(len, sizeof(double));
        D_pow[i][i] = pow(D[i][i], -0.5);
    }
    return D_pow;
}

/* Computes m^T, assumes m size is h*w */
double** transpose(double** m, int h, int w) {
    double **t; int i, j;
    t = (double**)calloc(w, sizeof(double*));
    for (i = 0; i < w; ++i) {
        t[i] = (double*)calloc(h, sizeof(double));
        for (j = 0; j < h; ++j) {
            t[i][j] = m[j][i];
        }
    }
    return t;
}

/* Computes m1*m2, assumes m1 size is n*m and m2 size is m*t */
double** matrix_mul(double** m1, double** m2, int n, int m, int t) {
    double** W; int i, j, k;
    W = (double**)calloc(n, sizeof(double*));
    for (i = 0; i < n; ++i) {
        W[i] = (double*)calloc(t, sizeof(double));
        for (j = 0; j < t; ++j) {
            for (k = 0; k < m; ++k)
                W[i][j] += m1[i][k] * m2[k][j];
        }
    }
    return W;
}

/* Computes m1*m2, assumes both m1 and m2 sizes are n*n */
double** sq_matrix_mul(double** m1, double** m2, int n) {
    return matrix_mul(m1, m2, n, n, n);
}

/* Computes the normalized similarity matrix */
double** graph_laplacian(double** A, double** D, int row_count) {
    double **D_pow, **mid, **W;
    D_pow = degree_matrix_minus_half_pow(D, row_count); /* (D^(-1/2)) */
    mid = sq_matrix_mul(D_pow, A, row_count); /* (D^(-1/2)) * A */
    W = sq_matrix_mul(mid, D_pow, row_count); /* (D^(-1/2)) * A * (D^(-1/2)) */
    free_matrix(mid, row_count);
    free_matrix(D_pow, row_count);
    return W;
}

double** norm_impl(double** X, int row_count, int row_len) {
    double **A, **D, **W;
    A = sym_impl(X, row_count, row_len); 
    D = degree_matrix(A, row_count);
    W = graph_laplacian(A, D, row_count);
    free_matrix(A, row_count);
    free_matrix(D, row_count);
    return W;
}

/* Computes (||H_next - H||_F)^2 */
double frobenius_norm_squared(double** h_next, double** h, int n, int k) {
    double norm = 0.0; int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            norm += pow(h_next[i][j] - h[i][j], 2);
        }
    }
    return norm;
}

/* W size is n*n, H size is n*k */
void update_H_next(double** H, double** H_next, double** W, int n, int k) {
    double **H_t, **WH, **HH_t, **HH_tH;
    int i, j;
    H_t = transpose(H, n, k); /* H^t */
    WH = matrix_mul(W, H, n, n, k); /* W*H */
    HH_t = matrix_mul(H, H_t, n, k, n); /* H*H^t */
    HH_tH = matrix_mul(HH_t, H, n, n, k); /* H*H^t*H */
    free_matrix(HH_t, n);
    free_matrix(H_t, k);
    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            H_next[i][j] = H_ij(H, WH, HH_tH, i, j);
        }
    }
    free_matrix(WH, n);
    free_matrix(HH_tH, n);
}

/* H <- H_next */
void advance_H(double **H, double **H_next, int n, int k) {
    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            H[i][j] = H_next[i][j];
        }
    }
}

/* symnmf algorithm implementation */
void optimize_H(double** H, double** W, int n, int k) {
    double **H_next, diff_frob; int i;
    H_next = (double**)calloc(n, sizeof(double*));
    for (i = 0; i < n; ++i)
        H_next[i] = (double*)calloc(k, sizeof(double));
    for (i = 0; i < max_iter; i++) {
        update_H_next(H, H_next, W, n, k);
        diff_frob = frobenius_norm_squared(H_next, H, n, k); /* (||H_next - H||_F)^2 */
        advance_H(H, H_next, n, k);
        if (diff_frob < eps) /* convergence */
            break;
    }
    free_matrix(H_next, n);
}

/* gets row count and row length */
void get_matrix_size(char *file_name, int *rows, int *columns) {
    double num; int ch;
    FILE *file;
    *rows = 0;
    *columns = 0;
    file = fopen(file_name, "r");
    while(fscanf(file, "%lf", &num) != EOF) {
        ch = fgetc(file);
        if(*rows == 0)
            (int)(*columns)++;
        if(ch == '\n' || ch == EOF)
            (int)(*rows)++;
    }
    fclose(file);
}

/* reads a file's content and converts to 2d array */
double** read_matrix(char *file_name, int *row_count, int *row_len) {
    FILE *file;
    double **data;
    int i, j, ch;
    get_matrix_size(file_name, row_count, row_len);
    file = fopen(file_name, "r");
    data = (double**)calloc(*row_count, sizeof(double*));
    for (i = 0; i < *row_count; ++i)
        data[i] = (double*)calloc(*row_len, sizeof(double));
    i = 0; j = 0;
    while(fscanf(file, "%lf", &(data[i][j])) != EOF) {
        ch = fgetc(file);
        if(ch == '\n') {
            i++;
            j = 0;
        } else
            j++;
    }
    fclose(file);
    return data;
}

/* prints a 2d array in the requsted format */
void print_matrix(double **matrix, int row_count, int row_len) {
    int i, j;
    for(i = 0; i < row_count; ++i) {
        for(j = 0; j < row_len; ++j) {
            printf("%.4f", matrix[i][j]);
            if (j != row_len - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    char *op, *file_name;
    double **input_matrix, **result;
    int row_count, row_len;

    op = argv[1];
    file_name = argv[2];
    if(argc != 3) {
        printf("an error has occured");
    }

    input_matrix = read_matrix(file_name, &row_count, &row_len);
    if(!strcmp(op, "sym")) {
        result = sym_impl(input_matrix, row_count, row_len);
    } else if (!strcmp(op, "ddg")) {
        result = ddg_impl(input_matrix, row_count, row_len);
    } else if (!strcmp(op, "norm")) {
        result = norm_impl(input_matrix, row_count, row_len);
    } else {
        printf("an error has occured");
        free_matrix(input_matrix, row_count);
        exit(1);
    }
    if (result) {
        print_matrix(result, row_count, row_count);
        free_matrix(result, row_count);
        free_matrix(input_matrix, row_count);
    }
    return 0;
}