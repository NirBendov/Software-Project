# ifndef SYMNMF_H_
# define SYMNMF_H_

/* frees a 2d array matrix[row_count][] */
void free_matrix(double** matrix, int row_count);

/* implements the sym algorithm, takes in X[row_count][row_num] */
double** sym_impl(double** X, int row_count, int row_len);

/* implements the ddg algorithm, takes in X[row_count][row_num] */
double** ddg_impl(double** X, int row_count, int row_len);

/* implements the norm algorithm, takes in X[row_count][row_num] */
double** norm_impl(double** X, int row_count, int row_len);

/* implements the symnmf algorithm, takes in H[n][k] and W[n][n] */
void optimize_H(double** H, double** W, int n, int k);

# endif