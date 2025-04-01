import numpy as np
import sys
import symnmfmodule
from kmeans import kmeans  # Import HW1 KMeans function
from sklearn.metrics import silhouette_score
np.random.seed(1234)


def analyze_symnmf(k, data):

    W = symnmfmodule.norm(data)
    if type(W) is str: #returned an error message
        print(W)
        exit()

    m = np.mean(W)

    # Perform SymNMF to compute the H matrix.
    initial_H = np.random.uniform(0, 2 * np.sqrt(m / k), size=(len(data), k))
    H = symnmfmodule.symnmf(initial_H.tolist(), W)
    if type(H) is str: #returned an error message
        print(H)
        exit()

    # Determine Cluster Labels by finding the index of the maximum value
    labels = np.argmax(H, axis=1)

    return silhouette_score(data, labels)

def analyze_kmeans(k, data):
    n, dims = len(data), len(data[0])

    # Call the kmeans function from HW1 with 300 iterations and epsilon = 0.001
    centroids = kmeans(data, n, dims, k, 300, 0.001)

    # To calculate silhouette score, we need to assign clusters based on centroids
    labels = []
    for point in data:
        min_dist = distance(point, centroids[0])
        label = 0
        for i in range(1, k):
            dist = distance(point, centroids[i])
            if dist < min_dist:
                min_dist = dist
                label = i
        labels.append(label)

    return silhouette_score(data, labels)


# Compute the Euclidean distance
def distance(a, b):
    return np.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def main():
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        return

    k = int(sys.argv[1])
    file_name = sys.argv[2]

    try:
        # Load the data with proper float conversion and error handling
        with open(file_name, "r") as file:
            data = [[float(x) for x in s.strip().split(",")] for s in file if s.strip()]
    except ValueError as e:
        print(f"An Error Has Occurred  {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An Error Has Occurred: {e}")
        sys.exit(1)

    nmf_score = analyze_symnmf(k, data)
    kmeans_score = analyze_kmeans(k, data)

    print(f'nmf: {"{:.4f}".format(float(nmf_score)) if not np.isnan(nmf_score) else "N/A"}')
    print(f'kmeans: {"{:.4f}".format(float(kmeans_score)) if not np.isnan(kmeans_score) else "N/A"}')


if __name__ == "__main__":
    main()