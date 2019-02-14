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
	Xmin = np.min(X_arr, axis = 0)
	# print Xmin
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

 
def execute_KNN_CV(K, X_norm, folds, distance_function):
	start_time = float(0.0)
	accuracy = float(0.0)
	X_norm = copy.deepcopy(X_norm)
	X_split = splitCV(X_norm, folds)

	for i in range(folds):
		X_test = np.copy(X_split[i])
		X_train = np.array([])
		for j in range(folds):
			if (i == j):
				continue
			if (X_train.size == 0):
				X_train = X_split[j]
			else:
				X_train = np.concatenate((X_train, X_split[j]))
		start_time = time.time()
		accuracy += KNN(K, X_train, X_test, distance_function)
		print accuracy
		end_time = time.time()
		time_taken = end_time - start_time

	return accuracy/folds, time_taken

def execute_KNN_TT(K, X_norm, percent, distance_function):
	X_norm = copy.deepcopy(X_norm)
	X_splitTT = splitTT(X_norm, percent)

	start_time = time.time()
	accuracy = KNN(K, X_splitTT[0], X_splitTT[1], distance_function)
	end_time = time.time()
	time_taken = end_time - start_time

	return accuracy, time_taken

def record_results(data):
	X_norm = copy.deepcopy(data)
	if os.path.isfile('results.txt'):
		os.remove('results.txt')
		header_results = ['Accuracy (%)', '\n0.7-0.3', 'Train and Test \n0.6-0.4', '\n0.5-0.5', '\n5 Folds', 'Cross Validation\n10 Folds', '\n15 Folds']
		header_time = ['Run Time(s)', '\n0.7-0.3', 'Train and Test\n0.6-0.4', '\n0.5-0.5', '\n5 Folds', 'Cross Validation\n10 Folds', '\n15 Folds']

		for i in range(3):
			if (i == 0):
				distance_function = euclidean
				distance_header = 'Euclidean Distance Function'

			elif(i == 1):
				distance_function = manhattan
				distance_header = 'Manhattan Distance Function'

			else:
				distance_function = minkowski
				distance_header = "Minkowski Distance Function"

	#K = 1
	results_K1 = ['K = 1']
	time_taken_K1 = ['K = 1']

	acc_K1, time_K1 = execute_KNN_TT(1, X_norm, 0.7, distance_function)
	results_K1.extend([round(acc_K1,2)])
	time_taken_K1.extend([round(time_K1,2)])
	acc_K1, time_K1 = execute_KNN_TT(1, X_norm, 0.6, distance_function)
	results_K1.extend([round(acc_K1,2)])
	time_taken_K1.extend([round(time_K1,2)])
	acc_K1, time_K1 = execute_KNN_TT(1, X_norm, 0.5, distance_function)
	results_K1.extend([round(acc_K1,2)])
	time_taken_K1.extend([round(time_K1,2)])

	acc_K1, time_K1 = Execute_KNN_CV(1, X_norm, 5, distance_function)
	results_K1.extend([round(acc_K1,2)])
	time_taken_K1.extend([round(time_K1,2)])
	acc_K1, time_K1 = Execute_KNN_CV(1, X_norm, 10, distance_function)
	results_K1.extend([round(acc_K1,2)])
	time_taken_K1.extend([round(time_K1,2)])
	acc_K1, time_K1 = Execute_KNN_CV(1, X_norm, 15, distance_function)
	results_K1.extend([round(acc_K1,2)])
	time_taken_K1.extend([round(time_K1,2)])

	#K = 5
	results_K5 = ['K = 5']
	time_taken_K5 = ['K = 5']

	acc_K5, time_K5 = execute_KNN_TT(5, X_norm, 0.7, distance_function)
	results_K5.extend([round(acc_K5,2)])
	time_taken_K5.extend([round(time_K5,2)])
	acc_K5, time_K5 = execute_KNN_TT(5, X_norm, 0.6, distance_function)
	results_K5.extend([round(acc_K5,2)])
	time_taken_K5.extend([round(time_K5,2)])
	acc_K5, time_K5 = execute_KNN_TT(5, X_norm, 0.5, distance_function)
	results_K5.extend([round(acc_K5,2)])
	time_taken_K5.extend([round(time_K5,2)])

 	acc_K5, time_K5 = execute_KNN_CV(5, X_norm, 5, distance_function)
	results_K5.extend([round(acc_K5,2)])
	time_taken_K5.extend([round(time_K5,2)])
	acc_K5, time_K5 = execute_KNN_CV(5, X_norm, 10, distance_function)
	results_K5.extend([round(acc_K5,2)])
	time_taken_K5.extend([round(time_K5,2)])
	acc_K5, time_K5 = execute_KNN_CV(5, X_norm, 15, distance_function)
	results_K5.extend([round(acc_K5,2)])
	time_taken_K5.extend([round(time_K5,2)])

	#K = 10
	results_K10 = ['K = 10']
	time_taken_K10 = ['K = 10']     

	acc_K10, time_K10 = execute_KNN_TT(10, X_norm, 0.7, distance_function)
	results_K10.extend([round(acc_K10,2)])
	time_taken_K10.extend([round(time_K10,2)])
	acc_K10, time_K10 = execute_KNN_TT(10, X_norm, 0.6, distance_function)
	results_K10.extend([round(acc_K10,2)])
	time_taken_K10.extend([round(time_K10,2)])
	acc_K10, time_K10 = execute_KNN_TT(10, X_norm, 0.5, distance_function)
	results_K10.extend([round(acc_K10,2)])
	time_taken_K10.extend([round(time_K10,2)])

	acc_K10, time_K10 = execute_KNN_CV(10, X_norm, 5, distance_function)
	results_K10.extend([round(acc_K10,2)])
	time_taken_K10.extend([round(time_K10,2)])
	acc_K10, time_K10 = execute_KNN_CV(10, X_norm, 10, distance_function)
	results_K10.extend([round(acc_K10,2)])
	time_taken_K10.extend([round(time_K10,2)])
	acc_K10, time_K10 = execute_KNN_CV(10, X_norm, 15, distance_function)
	results_K10.extend([round(acc_K10,2)])
	time_taken_K10.extend([round(time_K10,2)])

	#K = 15
	results_K15 = ['K = 15']
	time_taken_K15 = ['K = 15']

	acc_K15, time_K15 = execute_KNN_TT(15, X_norm, 0.7, distance_function)
	results_K15.extend([round(acc_K15,2)])
	time_taken_K15.extend([round(time_K15,2)])
	acc_K15, time_K15 = execute_KNN_TT(15, X_norm, 0.6, distance_function)
	results_K15.extend([round(acc_K15,2)])
	time_taken_K15.extend([round(time_K15,2)])
	acc_K15, time_K15 = execute_KNN_TT(15, X_norm, 0.5, distance_function)
	results_K15.extend([round(acc_K15,2)])
	time_taken_K15.extend([round(time_K15,2)])

	acc_K15, time_K15 = execute_KNN_CV(15, X_norm, 5, distance_function)
	results_K15.extend([round(acc_K15,2)])
	time_taken_K15.extend([round(time_K15,2)])
	acc_K15, time_K15 = execute_KNN_CV(15, X_norm, 10, distance_function)
	results_K15.extend([round(acc_K15,2)])
	time_taken_K15.extend([round(time_K15,2)])
	acc_K15, time_K15 = execute_KNN_CV(15, X_norm, 15, distance_function)
	results_K15.extend([round(acc_K15,2)])
	time_taken_K15.extend([round(time_K15,2)])

	#K = 20
	results_K20 = ['K = 20']
	time_taken_K20 = ['K = 20']     

	acc_K20, time_K20 = execute_KNN_TT(20, X_norm, 0.7, distance_function)
	results_K20.extend([round(acc_K20,2)])
	time_taken_K20.extend([round(time_K20,2)])
	acc_K20, time_K20 = execute_KNN_TT(20, X_norm, 0.6, distance_function)
	results_K20.extend([round(acc_K20,2)])
	time_taken_K20.extend([round(time_K20,2)])
	acc_K20, time_K20 = execute_KNN_TT(20, X_norm, 0.5, distance_function)
	results_K20.extend([round(acc_K20,2)])
	time_taken_K20.extend([round(time_K20,2)])

	acc_K20, time_K20 = execute_KNN_CV(20, X_norm, 5, distance_function)
	results_K20.extend([round(acc_K20,2)])
	time_taken_K20.extend([round(time_K20,2)])
	acc_K20, time_K20 = execute_KNN_CV(20, X_norm, 10, distance_function)
	results_K20.extend([round(acc_K20,2)])
	time_taken_K20.extend([round(time_K20,2)])
	acc_K20, time_K20 = execute_KNN_CV(20, X_norm, 15, distance_function)
	results_K20.extend([round(acc_K20,2)])
	time_taken_K20.extend([round(time_K20,2)])

	table_results = tabulate.tabulate([results_K1, results_K5, results_K10, results_K15, results_K20], headers = header_results, tablefmt = 'psql', numalign = 'center')
	table_time = tabulate.tabulate([time_taken_K1, time_taken_K5, time_taken_K10, time_taken_K15, time_taken_K20], headers = header_time, tablefmt = 'psql', numalign = 'center')


if __name__=='__main__':

	data = sys.argv[1]
	X_arr = loadData(data)

	X_norm = dataNorm(X_arr)

	# print testNorm([X_norm])
	# print testNorm([X_split])

	#Running these will execute the KNN code to print accuracy
	# distance_function = euclidean
	# execute_KNN_TT(5, X_norm, 0.5, distance_function)
	# results = execute_KNN_CV(5, X_norm, 5, distance_function)
	# print results

	#Running this will test the recording of results
	record_results(X_norm)





