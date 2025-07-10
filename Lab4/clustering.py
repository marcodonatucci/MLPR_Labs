import numpy
import scipy.cluster.vq
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import sklearn.metrics

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def load_iris():
    
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def k_means(D, n, seed=0):
    numpy.random.seed(0)
    centroids, clusterLabels = scipy.cluster.vq.kmeans2(D.T, n, minit='points', iter=100) # Compute the centroids and cluster labels. We transpose D to have a matrix of row-vector samples
    centroids = centroids.T # We transpose the centroids to have a matrix of column-vector centroids as output
    return centroids, clusterLabels

def plot_scatter(D, clusterLabels):
    plt.figure()
    for label in range(clusterLabels.max()+1): # Cluster labels range from from 0 up to <number of clusters> - 1 included. clusterLabels.max()+1 corresponds to the number of clusters.
        plt.plot(D[3, clusterLabels == label], D[2, clusterLabels == label], 'o') # We filter the samples based on the cluster labels
    plt.show()

def compute_distortion(D, centroids, clusterLabels):
    distortion = 0
    for i in range(D.shape[1]): # We iterate over all samples
        distortion += numpy.linalg.norm(D[:, i] - centroids[:, clusterLabels[i]])**2 # For a given sample D[:, i] (1D array) we retrieve the corresponding cluster labels clusterLabels[i]. The cluster label is also the index of the corresponding centroid in matrix centroids. We compute the norm of the difference (numpy.linalg.norm) and take its square. 
    return distortion / D.shape[1] # To get an average distortion, we divide by the number of samples

def hierarchical_clustering(D, method = 'single'):
    linkageMatrix = scipy.cluster.hierarchy.linkage(D.T, method = method, metric = 'euclidean') # Compute linkage matrix
    return linkageMatrix

def hierarchical_flatten_linkage_matrix(Z, nCluTarget):
    return scipy.cluster.hierarchy.fcluster(Z, nCluTarget, criterion = 'maxclust') # Cut the dendorgram at a level that gives nCluTarget clusters and return the correspondig cluster labels

def compute_silhouette_score(D, clusterLabels):
    return sklearn.metrics.silhouette_score(D.T, clusterLabels) # Silhouette score for the provided cluster labels

if __name__ == '__main__':

    D, L = load_iris()
    distortionValues = []
    for nClusters in [1,2,3,4,5,6]:
        centroids, clusterLabels = k_means(D, nClusters) # Compute k-means solution
        plot_scatter(D, clusterLabels) 
        distortionValues.append(compute_distortion(D, centroids, clusterLabels)) # Compute the distortion for the current solution
    plt.figure()
    plt.plot(numpy.arange(1, 7), numpy.array(distortionValues))
    plt.xlabel('number of clusters')
    plt.ylabel('distortion')
    plt.show()

    for method in ['single', 'complete', 'average']:
        linkageMatrix = scipy.cluster.hierarchy.linkage(D.T, method = method, metric = 'euclidean') # Compute the linkage matrix

        for nClusters in [2,3,4]: # For plots we consider 2, 3 and 4 cluster hypotheses
            clusterLabels = scipy.cluster.hierarchy.fcluster(linkageMatrix, nClusters, criterion = 'maxclust') # Compute flat clusters from the linkage matrix, where the number of clusters is given by nClusters
            plot_scatter(D, clusterLabels)

        silScores = []
        for nClusters in range(2, 13): # We consider a larger number of hypotheses for silhouette plots
            clusterLabels = scipy.cluster.hierarchy.fcluster(linkageMatrix, nClusters, criterion = 'maxclust')
            silhouette = sklearn.metrics.silhouette_score(D.T, clusterLabels)
            silScores.append(silhouette)
        plt.figure()
        plt.plot(numpy.arange(2, 13), numpy.array(silScores))
        plt.xlabel('number of clusters')
        plt.ylabel('silhouette score')
        plt.show()
        
        
