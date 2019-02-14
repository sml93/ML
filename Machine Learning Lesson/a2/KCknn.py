#!/usr/bin/python
import sys
import time
import copy
import math
import operator
import numpy as np
import a2 as driver

# Normalize Data Attributes
def dataNorm(X):
	X_norm = X
	for j in range(len(X_norm[0])-1):
		Xmax = X[:,j].max()
		Xmin = X[:,j].min()
		for i in range(len(X_norm)):
			data = X[i,j]
			X_norm[i, j] = (data - Xmin) / (Xmax - Xmin)
	return X_norm

def splitTT(X, percent):
	X_temp = X
	np.random.shuffle(X_temp)
	copyRange = int(len(X_temp) * percent)
	X_split = np.split(X_temp, [copyRange])
	return X_split

# Split Cross Validation into K Units
def splitCV(X_norm, k):
  np.random.shuffle(X_norm)
  X_split = np.array_split(X_norm, k)
  return X_split

def KNNManhattan(X_train, X_test, K):
  return KNN(X_train, X_test, K, manhattanDistance)

def KNNMinkow(X_train, X_test, K):
  return KNN(X_train, X_test, K, monkowskiDistance)

def manhattanDistance(x0, x1):
  distance = 0
  for x in range(len(x0)-1):
    distance += abs(x0[x] - x1[x])
  return distance

def minkowskiDistance(x0, x1):
  distance = 0
  for x in range(len(x0)-1):
    distance += pow(abs(x0[x] - x1[x]),3)
  oneOverP = 1.0/3.0
  return pow(distance, oneOverP)

def euclideanDistance(x0, x1):
  distance = 0
  for x in range(len(x0)-1):
    distance += pow((x0[x] - x1[x]), 2)
  return math.sqrt(distance)

def delete_list(list):
  for i in range(len(list)):
    list.remove(list[0])

def KNN(X_train, X_test, K, DistanceFunc = euclideanDistance):
    hits = 0

    for i in range(len(X_test)):
        distanceList = list()
        iRow = X_test[i]

        for j in range(len(X_train)):
            distanceList.append([j, DistanceFunc(iRow, X_train[j])])
        # Sort by Distance
	
        distanceList.sort(key = lambda x : x[1])
        neighbours = [item[0] for item in distanceList[:K]]
        results = [X_train[index][8] for index in neighbours]
	
        # Obtain Highest 'Vote'
        highestCount = np.bincount(results).argmax()

        if highestCount == int(X_test[i][8]):
            hits += 1

    # Return Accuracy
    return float(hits) / float(len(X_test))

def CrossValidateKNN(data, K, folds, distance_func = euclideanDistance):     
  accuracy = float(0.0)
  time_taken = float(0.0)
  
  X_norm = copy.deepcopy(data)
  X_split = splitCV(X_norm, folds)

  for i in range(folds):
      X_test = np.copy(X_split[i])
      X_train = np.array([])
      for j in range(folds):
          if (j == i):
              continue
          if (X_train.size == 0):
              X_train = X_split[j]
          else:
              X_train = np.concatenate((X_train, X_split[j]))    
      start_time = time.time()
      accuracy += KNN(X_train, X_test, K, distance_func)
      print(accuracy)
      end_time = time.time()
      time_taken += (end_time - start_time)

  return accuracy, time_taken

def ExecuteSplitTT(data, knn_function, K, percentage):
    X_norm = copy.deepcopy(data)
    X_split = splitTT(X_norm, percentage)

    start_time = time.time()
    accuracy   = knn_function(X_split[0], X_split[1], K) 
    end_time   = time.time()
    
    delete_list(X_split)
    np.delete(X_norm, X_norm[:])

    time_elapsed = end_time - start_time
    
    return accuracy, time_elapsed

def main():
  print "Using Data Source : ", sys.argv[1]
  X_data = driver.loadData(sys.argv[1])
  X_norm = dataNorm(X_data)

  # Test Data
  ExecuteSplitTT(X_norm, KNN, 5, 0.5)
  results = CrossValidateKNN(X_norm, 5, 5)

#  print(results)

if __name__ == "__main__":
    main() 
