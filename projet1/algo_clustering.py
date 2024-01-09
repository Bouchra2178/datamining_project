import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA

class KMeans:
    def __init__(self, k=3, max_iterations=100, distance_metric='euclidean'):
        self.k = k
        self.max_iterations = max_iterations
        self.distance_metric = distance_metric
        self.centroids = None
        self.clusters = None
  
    def fit(self, data):
        # Initialize centroids randomly
        # self.centroids = [data[i] for i in np.random.choice(len(data), self.k, replace=False)]
        self.centroids = [data[i] for i in np.random.choice(len(data), self.k, replace=False)]

        for _ in range(self.max_iterations):
            # Assign instances to clusters
            self.clusters = [[] for _ in range(self.k)]
            for instance in data:
                nearest_cluster_index = self.find_nearest_cluster(instance,self.centroids,self.distance_metric)
                self.clusters[nearest_cluster_index].append(instance) #Affecter chaque instances D[i] au
                                                             #groupe dont il est le plus proche de son centre;

            # Update centroids avec mean of cluster
            new_centroids = [self.calculate_centroid(cluster) if cluster else self.centroids[i] for i, cluster in enumerate(self.clusters)]

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids
        
        
    def find_nearest_cluster(self, instance, centroids, distance_metric):
        distances = [self.calculate_distance(instance, centroid, distance_metric) for centroid in centroids]
        return np.argmin(distances)
    
    def calculate_distance(self, instance1, instance2, distance_metric='euclidean'):
            if distance_metric == 'manhattan':
             return np.sum(np.abs(instance1 - instance2))
            elif distance_metric == 'euclidean':
                return np.sqrt(np.sum((instance1 - instance2) ** 2))
            elif distance_metric == 'minkowski':
             p = 2  # You can specify a different p if needed
             return np.power(np.sum(np.power(np.abs(instance1 - instance2), p)), 1/p)
            elif distance_metric == 'cosine':
                dot_product = np.dot(instance1, instance2)
                norm_instance1 = np.linalg.norm(instance1)
                norm_instance2 = np.linalg.norm(instance2)
                return 1 - dot_product / (norm_instance1 * norm_instance2)
            elif distance_metric == 'hamming':
                return np.sum(np.array(list(instance1)) != np.array(list(instance2)))
            else:
               raise ValueError("Invalid distance metric. Choose from 'euclidean', 'manhattan', 'minkowski', 'cosine', 'hamming'.")
    def calculate_centroid(self, cluster):
        # Calculate the centroid of a cluster
        return np.mean(cluster, axis=0)

    def intra_cluster_distance(self, cluster):
        # Calculate the average pairwise distance within a cluster
        distances = [self.calculate_distance(instance1, instance2, self.distance_metric)
                     for i, instance1 in enumerate(cluster)
                     for instance2 in cluster[i + 1:]]  # Avoid duplicate pairs
        return np.mean(distances) if distances else 0

    def inter_cluster_distance(self, cluster1, cluster2):
        # Calculate the average pairwise distance between two clusters
        distances = [self.calculate_distance(instance1, instance2, self.distance_metric)
                     for instance1 in cluster1
                     for instance2 in cluster2]
        return np.mean(distances) if distances else 0

    def evaluate_clustering(self):
        # Evaluate intra-cluster and inter-cluster distances for each cluster
        intra_cluster_distances = [self.intra_cluster_distance(cluster) for cluster in self.clusters]
        inter_cluster_distances = []
        for i, cluster1 in enumerate(self.clusters):
            for j, cluster2 in enumerate(self.clusters):
                if i != j:
                    inter_cluster_distances.append(self.inter_cluster_distance(cluster1, cluster2))

        # Return the average intra-cluster distance and the average inter-cluster distance
        return np.mean(intra_cluster_distances), np.mean(inter_cluster_distances)

    def plot_silhouette_scores(self, data, k_range):
        silhouette_scores = []
        for k in k_range:
            kmeans = KMeans(k=k)
            kmeans.fit(data)
            silhouette_avg = kmeans.calculate_silhouette_score(data)
            silhouette_scores.append(silhouette_avg)
        # Plot the results
        plt.plot(k_range, silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Different Values of k')
        plt.show()
        
    def calculate_silhouette_score(self, data):
        if self.centroids is None or self.clusters is None:
            raise ValueError("K-Means must be fitted before calculating silhouette score.")
        self.silhouette_avg = 0
        for i, cluster in enumerate(self.clusters):
            for instance in cluster:
                #r la distance moyenne du point à son groupe
                a_i = np.mean([self.calculate_distance(instance, other, self.distance_metric) 
                               for other in cluster if not np.array_equal(instance, other)])
                b_values = []
                for j, other_cluster in enumerate(self.clusters):
                    if i != j:
                        b_values.append(np.mean([self.calculate_distance(instance, other, self.distance_metric) for other in other_cluster]))

                b_i = min(b_values) if b_values else 0
                s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0
                self.silhouette_avg += s_i

        self.silhouette_avg /= len(data)
        return self.silhouette_avg
    
    def predict3(self, X):
        X_array = np.array(X)
        return np.argmin(np.linalg.norm(X_array[:, np.newaxis] - self.centroids, axis=-1), axis=-1)
    
    def predict(self, X):
        X_array = np.array(X)
        Result=[]
        for X in X_array:
            distances = self.distances(X)
            Result.append(np.argmin(distances, axis=0))
        
        return np.array(Result)
    def predict2(self, X):
        X_array = np.array(X)
        distances = self.distances(X_array)
        return np.argmin(distances, axis=0)
    def distances(self,X):
        distances = []
        for centroid in self.centroids:
            distance=self.calculate_distance(X, centroid, self.distance_metric)
            distances.append(distance)
            
        return np.array(distances)

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.silhouette_score = None
    def fit(self, data):
        self.labels = [0] * len(data)
        C = 0
        for P in range(len(data)):
            if not (self.labels[P] == 0):
                continue
            NeighborPts = self.region_query(data, P)
            if len(NeighborPts) < self.min_samples:
                self.labels[P] = -1
            else:
                C += 1
                self.grow_cluster(data, P, NeighborPts, C)

        return self.labels

    def grow_cluster(self, data, P, NeighborPts, C):
        self.labels[P] = C
        i = 0
        while i < len(NeighborPts):
            Pn = NeighborPts[i] 
            if self.labels[Pn] == 0:
                self.labels[Pn] = C
                PnNeighborPts = self.region_query(data, Pn)
                if len(PnNeighborPts) >= self.min_samples:
                    NeighborPts = np.concatenate((NeighborPts, PnNeighborPts))
            i += 1

    def region_query(self, data, P):
        neighbors = []
        for Pn in range(len(data)):
            if np.linalg.norm(data[P] - data[Pn]) < self.eps:
                neighbors.append(Pn)
        return neighbors
    
    # def grow_cluster(self, data, P, NeighborPts, C):
    #     self.labels[P] = C
    #     i = 0
    #     while i < len(NeighborPts):
    #         Pn = NeighborPts[i] 
    #         if self.labels[Pn] == 0:
    #             self.labels[Pn] = C
    #             PnNeighborPts = set(self.region_query(data, Pn))
    #             if len(PnNeighborPts) >= self.min_samples:
    #                 NeighborPts.update(PnNeighborPts)
    #         i += 1
    # def region_query(self, data, P):
    #     return [Pn for Pn in range(len(data)) if np.linalg.norm(data[P] - data[Pn]) < self.eps]

def performance2(X,label):
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    axes[0].scatter(x1, x2, c=X['Fertility'], cmap="viridis", alpha=0.7, s=100)
    axes[0].set_title('Before KMeans Clustering for all instances')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[1].scatter(x1, x2, c=label, cmap="viridis", alpha=0.7, s=100)
    axes[1].set_title('After KMeans Clustering for all instances')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    st.pyplot(fig)
   
def performance2(X,y,label):
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    y_flat = y.ravel()
    df = pd.DataFrame({'PC1': X_projected[:, 0], 'PC2': X_projected[:, 1], 'Label': 0})
    # Avant le clustering
    # fig_before = px.scatter(x=X_projected[:, 0], y=X_projected[:, 1], color=X['Fertility'], title='Before KMeans Clustering for all instances', labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'})
    fig_before=px.scatter(df, x='PC1', y='PC2', color='Label', title='Scatter Plot with PCA', labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
    fig_before.update_traces(marker=dict(size=10, opacity=0.7))
    df = pd.DataFrame({'PC1': X_projected[:, 0], 'PC2': X_projected[:, 1], 'Label': label.ravel()})
    # Après le clustering
    fig_after=px.scatter(df, x='PC1', y='PC2', color='Label', title='Scatter Plot with PCA', labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
    # fig_after = px.scatter(x=X_projected[:, 0], y=X_projected[:, 1], color=label, title='After KMeans Clustering for all instances', labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'})
    fig_after.update_traces(marker=dict(size=10, opacity=0.7))

    # Afficher les deux graphiques
    st.plotly_chart(fig_before)
    st.plotly_chart(fig_after) 
#metrci adjusted_rand
def adjusted_rand_index(true_labels, predicted_labels,algo):
    if algo=="DBSCAN":
        valid_indices = predicted_labels != -1
        predicted_labels = pd.Series(predicted_labels[valid_indices])
        true_labels = pd.Series(true_labels[valid_indices])
        predicted_labels.reset_index(drop=True, inplace=True)
        true_labels.reset_index(drop=True, inplace=True)
        predicted_labels=predicted_labels.values
        true_labels=true_labels.values
        predicted_labels-=1
    contingency_matrix = calculate_contingency_matrix(true_labels, predicted_labels)
    a = np.sum(np.square(contingency_matrix))
    b = np.sum(np.square(contingency_matrix.sum(axis=1)))
    c = np.sum(np.square(contingency_matrix.sum(axis=0)))
    d = np.sum(np.square(contingency_matrix.sum()))
    expected_index = (b * c) / d
    max_index = 0.5 * (b + c)
    return (a - expected_index) / (max_index - expected_index)

def calculate_contingency_matrix(true_labels, predicted_labels):
    unique_true_labels = np.unique(true_labels)
    unique_predicted_labels = np.unique(predicted_labels)
    contingency_matrix = np.zeros((len(unique_true_labels), len(unique_predicted_labels)))
    for i in range(len(true_labels)):
        true_label_index = np.where(unique_true_labels == true_labels[i])[0][0]
        predicted_label_index = np.where(unique_predicted_labels == predicted_labels[i])[0][0]
        contingency_matrix[true_label_index][predicted_label_index] += 1
    return contingency_matrix
#metric davies_bouldin
def davies_bouldin_index(data, labels,algo):
    # if algo=="DBSCAN":
    #     mask = (labels != -1)
    #     labels = labels[mask]
    #     labels-=1
    labels = np.where(labels == -1, 0, labels)
    num_clusters = len(np.unique(labels))
    cluster_centers = calculate_cluster_centers(data, labels, num_clusters)
    cluster_distances = calculate_cluster_distances(data, labels, cluster_centers, num_clusters)
    cluster_diameters = calculate_cluster_diameters(data, labels, cluster_centers, num_clusters)

    dbi_values = []

    for i in range(num_clusters):
        if i != np.argmax(cluster_distances[i]):
            dbi = (cluster_diameters[i] + cluster_diameters[np.argmax(cluster_distances[i])]) / cluster_distances[i][np.argmax(cluster_distances[i])]
            dbi_values.append(dbi)

    return np.mean(dbi_values)

def calculate_cluster_centers(data, labels, num_clusters):
    cluster_centers = []
    for i in range(num_clusters):
        cluster_data = data[labels == i]
        center = np.mean(cluster_data, axis=0)
        cluster_centers.append(center)
    return np.array(cluster_centers)

def calculate_cluster_distances(data, labels, cluster_centers, num_clusters):

    cluster_distances = np.zeros((num_clusters, num_clusters))

    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j:
                distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                cluster_distances[i][j] = distance

    return cluster_distances

def calculate_cluster_diameters(data, labels, cluster_centers, num_clusters):

    cluster_diameters = np.zeros(num_clusters)
    from sklearn.metrics import pairwise_distances
    for i in range(num_clusters):
        cluster_data = data[labels == i]
        distances = pairwise_distances(cluster_data, [cluster_centers[i]])
        diameter = np.max(distances)
        cluster_diameters[i] = diameter

    return cluster_diameters

def calinski_harabasz_index(X, labels):

    centroids = np.array([np.mean(X[labels == label], axis=0) for label in np.unique(labels)])
    overall_mean = np.mean(X, axis=0)
    n = len(X)
    k = len(centroids)
    between_cluster_variance = sum([len(X[labels == label]) * np.sum((centroid - overall_mean) ** 2) for label, centroid in zip(np.unique(labels), centroids)])
    within_cluster_variance = sum([np.sum((x - centroids[labels[i]]) ** 2) for i, x in enumerate(X)])

    score = (between_cluster_variance / (k - 1)) / (within_cluster_variance / (n - k))

    return score

def inter_cluster_variance(X, labels):

    centroids = np.array([np.mean(X[labels == label], axis=0) for label in np.unique(labels)])
    overall_mean = np.mean(X, axis=0)

    variance = sum([len(X[labels == label]) * np.sum((centroid - overall_mean) ** 2) for label, centroid in zip(np.unique(labels), centroids)])

    return variance

def intra_cluster_variance(X, labels):

    centroids = np.array([np.mean(X[labels == label], axis=0) for label in np.unique(labels)])

    variance = sum([np.sum((x - centroids[labels[i]]) ** 2) for i, x in enumerate(X)])

    return variance

def modele_Evaluation(Model,X1,X2,X3,X4):
    data = {
    'Algo': [Model],
    'Adjusted Rand Score': [X1],
    'Silhouette Score': [X2],
    'Davies Bouldin Score': [X3],
    'Calinski Harabasz Score': [X4]
    }
    df = pd.DataFrame(data)
    return df
    
