import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X = np.load("data/ex7_X.npy")
    return X

def draw_line(p1, p2, style="-k", linewidth=1):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

def plot_data_points(X, idx):
    # plots data points in X, coloring them so that those with the same
    # index assignments in idx have the same color
    plt.scatter(X[:, 0], X[:, 1], c=idx)
    
def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    # Plot the examples
    plot_data_points(X, idx)
    
    # Plot the centroids as black 'x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)
    
    # Plot history of the centroids with lines
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    
    plt.title("Iteration number %d" %i)

def draw_k_means(X, y_predict, predict_label, centroids):

    fig = plt.figure()
    label0 = plt.scatter(X.iloc[:, 0][y_predict==0], X.iloc[:, 1][y_predict==0])
    label1 = plt.scatter(X.iloc[:, 0][y_predict==1], X.iloc[:, 1][y_predict==1])

    plt.title(predict_label)
    plt.xlabel("V1")
    plt.ylabel("V2")

    plt.legend((label0, label1), ("label0", "label1"))
    # plot the center
    # :,0 : 去获取第一维所有的行， 0 获取第一列
    plt.scatter(centroids[:,0], centroids[:,1])

    plt.show()