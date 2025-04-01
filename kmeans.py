import sys
import math

def validate_k(k, n):
    if k <= 1 or k >= n:
        raise Exception("An Error Has Occurred")

def validate_iter(itr):
    if itr <= 1 or itr >= 1000:
        raise Exception("An Error Has Occurred")

def calculate_entry(vects, i):
    return sum(v[i] for v in vects) / len(vects)

def calculate_centroid(vects):
    return [calculate_entry(vects, i) for i in range(len(vects[0]))]

def distance(x, y):
    return math.sqrt(sum((x[i] - y[i])**2 for i in range(len(x))))

def epsilon_dists(curr_cents, next_cents, eps=0.001):
    return all(distance(curr_cents[i], next_cents[i]) < eps for i in range(len(curr_cents)))

def kmeans(lines, n, dims, k, max_iter, epsilon):
    validate_k(k, n)
    validate_iter(max_iter)

    # Initialize centroids
    centroids = [lines[i].copy() for i in range(k)]

    for t in range(max_iter):
        cent_to_vects = [[] for _ in range(k)]
        curr_cents = centroids.copy()

        # Assign points to nearest centroids
        for vect in lines:
            distances = [distance(vect, centroids[i]) for i in range(k)]
            index = distances.index(min(distances))
            cent_to_vects[index].append(vect)

        # Recompute centroids
        for j in range(k):
            if cent_to_vects[j]:  # Avoid division by zero
                centroids[j] = calculate_centroid(cent_to_vects[j])

        # Check convergence
        if epsilon_dists(curr_cents, centroids, eps=epsilon):
            break

    return centroids

def main():
    try:
        if len(sys.argv) < 3:
            raise Exception("An Error Has Occurred")

        k = int(sys.argv[1])
        txt = sys.argv[2]
        max_iter = 200
        if len(sys.argv) == 4:
            max_iter = int(sys.argv[2])
            txt = sys.argv[3]

        with open(txt, "r") as file:
            lines = [[float(x) for x in s.split(",")] for s in file if s.strip()]

        n, dims = len(lines), len(lines[0])

        centroids = kmeans(lines, n, dims, k, max_iter, 0.001)
        for cent in centroids:
            print(','.join(['{:.4f}'.format(elem) for elem in cent]))

    except Exception as e:
        print('An Error Has Occurred:', e)

if __name__ == '__main__':
    main()
