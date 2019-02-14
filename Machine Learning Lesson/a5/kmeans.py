import numpy as np
import matplotlib.pyplot as plt

import sys
import math
import copy

def loadData(filename):

    X = []
    count = 0
    text_file = open(filename, 'r')
    lines = text_file.readlines()

    for line in lines:

	X.append([])
	words = line.split('	')

	for word in words:

	    X[count].append(float(word))

	count += 1

    X_zeros = np.zeros((len(X), 1))
    X = np.hstack((X, X_zeros))

    return np.asarray(X)

def plotData(X):

    plt.figure('25/08/2010 Lightning Strike Record')
    plt.scatter(X[:, 0], X[:, 1], marker = 'x', c = 'blue')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.show(block = False)

def errCompute(X, M):

    cost = 0.0

    for row in X:
        
	cluster_id = int(row[2])
        cost += np.sqrt(np.sum((row[0] - M[cluster_id][0]) ** 2 + (row[1] - M[cluster_id][1]) ** 2))

    return cost / len(X)

def Group(X, M):

    for row in X:

        distance = (row[0] - M[:, 0]) ** 2 + (row[1] - M[:, 1]) ** 2
        closest_cluster = np.argmin(distance, axis = 0)
        row[2] = int(closest_cluster)

    return X

def calcMean(X, M):

    new_M = np.zeros((K_Clusters, X.shape[1] - 1))
    cluster_size = np.zeros((K_Clusters, 1))

    for row in X:

	cluster_id = int(row[2])
	new_M[cluster_id] += row[0:2]
	cluster_size[cluster_id] += 1
    
    return new_M / cluster_size

def find_Optimal_Clusters(X, M):

    old_M = copy.deepcopy(M)
    plt.figure('Cluster Overview')
    error_history = []
    iters_history = []
    iters = 0

    while True:

	new_X = Group(X, old_M)
	M = calcMean(new_X, old_M)

	if ((old_M == M).all()):

	    break

	old_M = copy.deepcopy(M)
	cost = errCompute(new_X, old_M)
	iters += 1
	error_history.append(cost)
        iters_history.append(iters)

	plt.clf()
        plt.title('Cluster Overview')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.scatter(new_X[:, 0], new_X[:, 1], c = new_X[:, 2], s = 50, cmap = 'viridis', alpha = 0.7, edgecolors = 'black')
        plt.scatter(old_M[:, 0], old_M[:, 1], marker = '*', c = 'red', s = 200)
	plt.pause(0.05)    

    plt.show(block = False)
	
    return new_X, M, error_history, iters_history

if __name__=='__main__':

    global K_Clusters

    K_Clusters = int(sys.argv[2])
    X = loadData(sys.argv[1])
    plotData(X) 

    M = np.copy(X[0: K_Clusters, 0: X.shape[1] - 1])
    new_X, M, error_history, iters_history = find_Optimal_Clusters(X, M)

    plt.figure('Error Plot')
    plt.title('Error Vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.plot(iters_history, error_history)
    plt.show()

    #X = Group(X, M)
    #new_M = calcMean(X, M)
    #cost = errCompute(X, np.array([[0, 0]]))
    














   
