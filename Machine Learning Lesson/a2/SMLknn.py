#!/usr/bin/python

from a2 import loadData, testNorm

import numpy as np
import math
import copy 	#For collections that are mutable or contain mutable items
import time 	#Required to include time module
import sys 		#Required to use system-specific parameters and functions
import operator #Exports a set of efficient functions corresponding to the intrinsic operators of Python
import os 		#Allows you to interface with the underlying operating system that Python is running on

def dataNorm(X_arr):
	X_norm = []
	X_output = []
	count = 0

	#Find the min and max of each col attributes
	Xmin = np.min(X_arr)#, axis = 0)
	print Xmin
	Xmax = np.max(X_arr, axis = 0)

	for row in X_arr:
		X_norm.append([])
		for i in range(8):
			#For each attribute, calculate normalised data using formula '(data - min)/(max - min)'
			col_norm = (X_arr[count, i] - Xmin[i]) / (Xmax[i] - Xmin[i])
			X_norm[count].append(col_norm)
			X_norm[count].append(X_arr.item((count, 8)))
			count += 1
	return np.asarray(X_norm)

def splitTT(X_norm, percent):
	np.random.shuffle(X_norm)
	split = int(len(X_norm)*percent)
	X_split = np.split(X_norm, [split])
	return X_split

#Split Cross Validation into K Units
def splitCV(X_norm, k):
	np.random.shuffle(X_norm)
	X_split = np.array_split(X_norm, k)
	return X_split

def euclidean(x0, x1):
	dist = 0 
	for x in range(len(x0)-1):
		dist += pow(x0[x] - x1[x], 2)
	return np.sqrt(dist)

def manhattan(x0, x1):
	dist = 0
	for x in range(len(x0)-1):
		dist += abs(x0[x] - x1[x])
	return dist

def minkowski(x0, x1):
	dist = 0
	p = 3.0
	for x in range(len(x0)-1):
		dist += pow(abs(x0[x]-x1[x]),int(p))
	oneOverP = 1.0/p
	return pow(dist, oneOverP)

def KNN(K, X_train, X_test, distance_function):
	correct = 0.0
	for i in range(len(X_test)):
		dist = []
		for j in range(len(X_train)):
			
			#Calculate distance between each X_test point with every X_train points
			dist.append([j, distance_function(X_test[i], X_train[j])])
			
		#Sort the distance from the shortest to the furthest
		dist.sort(key = operator.itemgetter(1))
			
		#Save the dist values based on K
		neighbours = [item[0] for item in dist[:K]]

		#Retrieve the output attribute of the K-nearest points in X_train
		results = [X_train[index][8] for index in neighbours]

	# Return the X_train point that have the most number of occurences
	predictions = np.bincount(neighbours).argmax()

	if predictions == int(X_test[i][8]):
		correct += 1

	return (correct / float(len(X_test))) * 100

 






