import numpy as np
import pandas as pd

from k_means import k_means


def load_iris():
    data = pd.read_csv("data/iris.data", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    print(data)
    classes = data["class"].to_numpy()
    features = data.drop("class", axis=1).to_numpy()
    return features, classes


def evaluate(clusters, labels):
    unique_clusters = np.unique(clusters, axis=0)
    for cluster in unique_clusters:
        labels_in_cluster = np.array([])
        for i in range(len(clusters)):
            if np.equal(clusters[i], cluster).all():
                labels_in_cluster = np.append(labels_in_cluster, labels[i])
        print(f"Cluster: {cluster}")
        for label_type in np.unique(labels):
            print(f"Num of {label_type}: {np.sum(labels_in_cluster == label_type)}")


def clustering(kmeans_pp):
    data = load_iris()
    features, classes = data
    intra_class_variance = []
    for i in range(100):
        assignments, centroids, error = k_means(features, 3, kmeans_pp)
        evaluate(assignments, classes)
        intra_class_variance.append(error)
    print(f"Mean intra-class variance: {np.mean(intra_class_variance)}")


if __name__ == "__main__":
    # clustering(kmeans_pp=False)
    clustering(kmeans_pp=True)
