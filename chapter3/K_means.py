import numpy as np
import pandas as pd
from utils import *

class KMeans_My:
    def __init__(self, n_cluster, random_state):
        self.n_cluster = n_cluster
        self.random_state = random_state
        self.idx = None
        self.centroids = None

    def _find_closest_centroids(self, X, centroids):
        """
        """
        m = X.shape[0]
        k = centroids.shape[0]

        # record the index of centroid for each x
        idx = np.zeros(m, dtype=int)

        for i in range(m):
            distances = []
            for j in range(k):
                norm_ij = np.linalg.norm(X[i] - centroids[j])
                distances.append(norm_ij)
            idx[i] = np.argmin(distances)
        
        return idx

    def compute_centroids(self, X, idx, K):
        """
          Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
        """
        m, n = X.shape

        centroids = np.zeros((K, n))

        for k in range(K):
            # points assigned to this centroid
            points = X[idx == k]
            centroids[k] = np.mean(points, axis=0)
        
        return centroids
    
    def _init_centroids(self, X, K):
        """
        
        """

        # Randomly reorder the indices of examples
        rng = np.random.default_rng(self.random_state)
        randidx = rng.permutation(X.shape[0])

        centroids = X[randidx[:K]]

        return centroids


    def run_kMeans(self, X, initial_centroids, max_iters = 10, plot_progress = False):
        """
        
        """
        # initialize values
        centroids = initial_centroids
        previous_centroids = centroids
        K = self.n_cluster

        for i in range(max_iters):

            print("K-Means iteration %d /%d" % (i, max_iters - 1))

            idx = self._find_closest_centroids(X, initial_centroids)

            # Optionally plot progress
            if plot_progress:
                plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
                # plt.pause(1) # 等待， 防止计算太快，图像没有刷新
                previous_centroids = centroids

            centroids = self.compute_centroids(X, idx, K)

        plt.show() 
        return centroids, idx

        
    def fit(self, X):
        """
        Train the model
        
        Args:
            X (ndarray):   (m, n) Data points
        """
        # Initialize the variables
        X = X.to_numpy()
        initial_centroids = self._init_centroids(X, self.n_cluster)
        iters = 10
        

        self.centroids, self.idx = self.run_kMeans(X, initial_centroids, iters, True)
    
    def predict(self, X):
        m = X.shape[0]
        y_predict = []

        for i in range(m):
            y_predict.append(self.idx[i])
        
        return y_predict
    