import math

import numpy as np


def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    return data[np.random.choice(len(data), k)]


def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    first_random = data[np.random.choice(len(data), 1)]
    centroids = np.array(first_random)
    for _ in range(k - 1):
        min_dists = np.array([])
        for point in data:
            min_dist = math.inf
            for centroid in centroids:
                dist = np.sqrt(np.sum(np.power((centroid - point), 2)))
                if dist < min_dist:
                    min_dist = dist
            min_dists = np.append(min_dists, min_dist)
        centroids = np.append(centroids, data[np.argmax(min_dists)]).reshape((_ + 2, 4))
    return centroids


def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point
    assignments = np.array([])
    for point in data:
        min_dist = math.inf
        current_centroid = None
        for centroid in centroids:
            dist = np.sqrt(np.sum(np.power((centroid - point), 2)))
            if dist < min_dist:
                min_dist = dist
                current_centroid = centroid
        assignments = np.append(assignments, current_centroid)
    assignments = assignments.reshape(data.shape)
    return assignments


def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    centroids, frequency = np.unique(assignments, return_counts=True, axis=0)
    new_centroids = np.array([])
    for j, centroid in enumerate(centroids):
        this_cent = np.array([])
        for i, assignment in enumerate(assignments):
            if np.array_equal(centroid, assignment):
                this_cent = np.append(this_cent, data[i])
        this_cent = this_cent.reshape((frequency[j], 4))
        this_cent = np.sum(this_cent, axis=0)
        this_cent /= frequency[j]
        new_centroids = np.append(new_centroids, this_cent)
    new_shape = (new_centroids.size // 4, 4)
    return new_centroids.reshape(new_shape)


def mean_intra_distance(data, assignments):
    return np.sqrt(np.sum(np.power((data - assignments), 2)))


def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100):  # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments)
