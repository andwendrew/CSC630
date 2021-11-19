import math
import numpy as np

def distance(X, Y):
    sum = 0
    for i in range (0, len(X)):
        sum = sum + (X[i] - Y[i])**2    
    sum = np.sqrt(sum)
    return sum  

def intracluster(X):
    n, d = np.shape(X)
    
    centroid = np.zeros(d)
    for i in range(0, n):
        centroid = centroid + X[i]
    centroid = centroid * 1/n

    sum = 0
    for i in range(0, n):
        sum = sum + distance(centroid, X[i])
    return sum

def intercluster(X, Y):
    n_X, d_X = np.shape(X)
    n_Y, d_Y = np.shape(Y)
    
    centroid_X = np.zeros(d_X)
    for i in range(0, n_X):
        centroid_X = centroid_X + X[i]
    centroid_X = centroid_X * 1/n_X
    
    centroid_Y = np.zeros(d_Y)
    for i in range(0, n_Y):
        centroid_Y = centroid_Y + Y[i]
    centroid_Y = centroid_Y * 1/n_Y
    
    return distance(centroid_X, centroid_Y)

def dunn_index(X):
    n = len(X)
    
    maxindex = 0
    for i in range (0, n):
        if intracluster(X[i]) > intracluster(X[maxindex]):
            maxindex = i
    
    mindex_x = 0
    mindex_y = 1
    for i in range(0, n):
        for j in range(0, i):
            if intercluster(X[i], X[j]) < intercluster(X[mindex_x], X[mindex_y]):
                mindex_x = i
                mindex_y = j
                
    ans = intercluster(X[mindex_x], X[mindex_y]) / intracluster(X[maxindex])
    return ans

def closest_index(x, X): # takes in array x that is a point, and array X of points, and returns the index of the point in X that is closest to x
    n = len(X)
    index = 0
    for i in range(0, n):
        if distance(x, X[i]) < distance(x, X[index]):
            index = i
    return index
        

class k_means_clustering():
    
    def __init__(self) :        
        pass
        
    def fit(self, X, k): # fit to a dataset X to k clusters
        n, d = np.shape(X) # n counts the number of data points in X, and d is dimension of the data points in X
        
        centroids = [] # create the centroids array 
        clusters = [] # create the set of clusters
        
        for i in range(0, k):
            clusters.append([]) # here, we are creating the k clusters and setting them to initially be empty
            centroids.append(X[i]) # here, we just set the initial array of centroids to be the first k points. 
        
        while True:              
            print(centroids)
            for i in range(0, n): # assigning each point to a cluster based on the centroid it is closest to
                clusters[closest_index(X[i], centroids)].append(X[i]) # for each X[i] in X, get index of centroid it is closest to, and add X[i]
                
            clustering_initial = []
            for i in range(0, k):
                clustering_initial.append(clusters[i])
            print(clustering_initial)
            
            for i in range(0, k): # get new centroids
                centroid = np.zeros(d)
                for j in range(0, len(clusters[i])):
                    centroid = centroid + clusters[i][j]
                centroid = centroid * 1/len(clusters[i])
                centroids[i] = centroid 
            
            for i in range(0, k): # reset the clusters
                clusters[i] = []
            
            for i in range(0, n): # get new clustering from new centroids
                clusters[closest_index(X[i], centroids)].append(X[i]) # for each X[i] in X, get index of centroid it is closest to, and add X[i]
           
            if clusters == clustering_initial: # break out of loop if upon update nothing changes
                break
            
            for i in range(0, k): # reset the clusters
                clusters[i] = []

        
# the fit method successfully implements the logic behind the k-means algorithm    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
  