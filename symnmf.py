import numpy as np
import sys
import symnmfmodule  # Import the C extension module

# Set random seed
np.random.seed(1234)

def read_file(file_name):
    """Reads the data from a file and returns it as a NumPy array."""
    try:
        file = open(file_name, "r")
        data = [[float(x) for x in s.split(",")] for s in file.readlines() if s != ""]
        file.close()
    except Exception as e:
        print("An Error Has Occurred")
        sys.exit(1)
    return data

def compute_symnmf(k, data):
    """Computes Symmetric Non-negative Matrix Factorization (SymNMF) and returns the factorization matrix H."""
    # Compute the normalized similarity matrix W
    W = symnmfmodule.norm(data)
    if type(W) is str:
        return W
    m = np.mean(W)
    N = len(W)        # Number of data points (rows in W)

    # Initialize H with random values
    H = np.random.uniform(0, 2 * np.sqrt(m / k), (N, k)).tolist()

    # Perform SymNMF
    H_final = symnmfmodule.symnmf(H, W)
    return H_final

def print_result(H):
    if type(H) is str:
        print(H)
    else:
        np.savetxt(sys.stdout, H, delimiter=',', fmt='%.4f')


def main():
    """Main function to handle command line arguments and perform the requested operation."""
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        return

    try:
        k = int(sys.argv[1])  # Number of clusters
        goal = sys.argv[2]    # Task to perform
        file_name = sys.argv[3]  # Path to the input file
    except ValueError:
        print("An Error Has Occurred")
        return

    # Read data from file
    data = read_file(file_name)
    if len(data) <= k:
        print("An Error Has Occurred")
        return

    # Perform the requested operation based on the goal
    if goal == 'symnmf':
        H = compute_symnmf(k, data)
        print_result(H)

    elif goal == 'sym':
        A = symnmfmodule.sym(data)
        print_result(A)

    elif goal == 'ddg':
        D = symnmfmodule.ddg(data)
        print_result(D)

    elif goal == 'norm':
        W = symnmfmodule.norm(data)
        print_result(W)

    else:
        print("An Error Has Occurred")

if __name__ == "__main__":
    main()