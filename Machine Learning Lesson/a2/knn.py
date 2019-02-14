from a2 import loadData, testNorm

import numpy as np
import math
import operator
import copy
import sys
import time
import tabulate
import os

def dataNorm(X_array):
  
    X_norm = []
    X_output = [] 
    count = 0

    # Find the minimum and maximum of each column attributes
    X_min = np.min(X_array, axis = 0)
    X_max = np.max(X_array, axis = 0)

    for row in X_array:
      
        X_norm.append([])
        
        for i in range(8):
	
	    # For each attribute, calculate normalized data using formula '(data - min)/(max - min)'
       	    col_norm = (X_array[count, i] - X_min[i]) / (X_max[i] - X_min[i])
	    X_norm[count].append(col_norm)

	X_norm[count].append(X_array.item((count, 8)))
        count += 1

    return np.asarray(X_norm)

def splitTT(percent, X_norm):	
    
    np.random.shuffle(X_norm)
    split = int(len(X_norm) * percent)
    X_split = np.split(X_norm, [split])
    
    return X_split

def splitCV(k, X_norm):

    np.random.shuffle(X_norm)
    X_split = np.array_split(X_norm, k)
    
    return X_split

def euclidean(instance1, instance2):

    distance = 0
    
    for i in range(len(instance1) - 1):

	distance += pow((instance1[i] - instance2[i]), 2)

    return math.sqrt(distance)

def manhattan(instance1, instance2):

    distance = 0

    for i in range(len(instance1) - 1):

        distance += abs(instance1[i] - instance2[i])

    return distance

def minkowski(instance1, instance2):

    distance = 0

    for i in range(len(instance1) - 1):

	distance += pow(abs(instance1[i] - instance2[i]), 3)

    return pow(distance, (1.0/3.0))

def KNN(K, X_train, X_test, distance_function):

    correct = float(0)

    for y in range(len(X_test)):

	distances = []
        
	for i in range(len(X_train)):

	    # Calculate distance between each X_test point with every X_train points
	    distances.append([i, distance_function(X_test[y], X_train[i])])	    

	# Sort the distances from shortest to furthest
    	distances.sort(key = operator.itemgetter(1))
	# Save the distance values based on K 
        neighbours = [item[0] for item in distances[:K]]
	# Retrieve the output attribute of the K-nearest points in X_train
        results = [X_train[index][8] for index in neighbours]
	# Return the X_train point that have the most number of occurences
	predictions = np.bincount(neighbours).argmax()

	# If prediction is correct
	if predictions == int(X_test[y][8]):

	    correct += 1

    return (correct / float(len(X_test))) * 100

def Execute_KNN_CV(K, X_norm, folds, distance_function):

    X_norm_temp = copy.deepcopy(X_norm)
    X_splitCV = splitCV(folds, X_norm_temp)
    Acc_CV = float(0)
    start_time = float(0)

    for j in range(folds):

	X_test = np.copy(X_splitCV[j])
	X_train = np.array([])

	for l in range(folds):

	    if (j == l):
		
		continue
		
            if (X_train.size == 0):

		X_train = X_splitCV[l]

	    else:
		
		X_train = np.concatenate((X_train, X_splitCV[l]))
        start_time = time.time()
	Acc_CV += KNN(K, X_train, X_test, distance_function)

    end_time = time.time()
    time_taken = end_time - start_time

    return Acc_CV / folds, time_taken

def Execute_KNN_TT(K, X_norm, percent, distance_function):

    X_norm_temp = copy.deepcopy(X_norm)
    X_splitTT = splitTT(percent, X_norm)

    start_time = time.time()
    Acc_TT = KNN(K, X_splitTT[0], X_splitTT[1], distance_function)
    end_time = time.time()
    time_taken = end_time - start_time

    return Acc_TT, time_taken

def Record_Results(data):

    X_norm = copy.deepcopy(data)

    if os.path.isfile('results.txt'):
    
	os.remove('results.txt')
    
    header_results = ['Accuracy (%)', '\n0.7-0.3', 'Train and Test\n0.6-0.4', '\n0.5-0.5', '\n5 Folds', 'Cross Validation\n10 Folds', '\n15 Folds']
    header_time = ['Run Time (s)', '\n0.7-0.3', 'Train and Test\n0.6-0.4', '\n0.5-0.5', '\n5 Folds', 'Cross Validation\n10 Folds', '\n15 Folds']    

    for i in range(3):

	if (i == 0):

	    distance_func = euclidean
	    distance_header = 'Euclidean Distance Function'

	elif(i == 1):

	    distance_func = manhattan
	    distance_header = 'Manhattan Distance Function'

	else:
	    distance_func = minkowski
	    distance_header = 'Minkowski Distance Function'


	#K = 1
	results_K1 = ['K = 1']
	time_taken_K1 = ['K = 1']     

	acc_K1, time_K1 = Execute_KNN_TT(1, X_norm, 0.7, distance_func)
	results_K1.extend([round(acc_K1,2)])
	time_taken_K1.extend([round(time_K1,2)])
	acc_K1, time_K1 = Execute_KNN_TT(1, X_norm, 0.6, distance_func)
	results_K1.extend([round(acc_K1,2)])
	time_taken_K1.extend([round(time_K1,2)])
	acc_K1, time_K1 = Execute_KNN_TT(1, X_norm, 0.5, distance_func)
	results_K1.extend([round(acc_K1,2)])
	time_taken_K1.extend([round(time_K1,2)])

	acc_K1, time_K1 = Execute_KNN_CV(1, X_norm, 5, distance_func)
	results_K1.extend([round(acc_K1,2)])
	time_taken_K1.extend([round(time_K1,2)])
	acc_K1, time_K1 = Execute_KNN_CV(1, X_norm, 10, distance_func)
	results_K1.extend([round(acc_K1,2)])
	time_taken_K1.extend([round(time_K1,2)])
	acc_K1, time_K1 = Execute_KNN_CV(1, X_norm, 15, distance_func)
	results_K1.extend([round(acc_K1,2)])
	time_taken_K1.extend([round(time_K1,2)])

	#K = 5
	results_K5 = ['K = 5']
	time_taken_K5 = ['K = 5']

	acc_K5, time_K5 = Execute_KNN_TT(5, X_norm, 0.7, distance_func)
	results_K5.extend([round(acc_K5,2)])
	time_taken_K5.extend([round(time_K5,2)])
	acc_K5, time_K5 = Execute_KNN_TT(5, X_norm, 0.6, distance_func)
	results_K5.extend([round(acc_K5,2)])
	time_taken_K5.extend([round(time_K5,2)])
	acc_K5, time_K5 = Execute_KNN_TT(5, X_norm, 0.5, distance_func)
	results_K5.extend([round(acc_K5,2)])
	time_taken_K5.extend([round(time_K5,2)])

 	acc_K5, time_K5 = Execute_KNN_CV(5, X_norm, 5, distance_func)
	results_K5.extend([round(acc_K5,2)])
	time_taken_K5.extend([round(time_K5,2)])
	acc_K5, time_K5 = Execute_KNN_CV(5, X_norm, 10, distance_func)
	results_K5.extend([round(acc_K5,2)])
	time_taken_K5.extend([round(time_K5,2)])
	acc_K5, time_K5 = Execute_KNN_CV(5, X_norm, 15, distance_func)
	results_K5.extend([round(acc_K5,2)])
	time_taken_K5.extend([round(time_K5,2)])

	#K = 10
	results_K10 = ['K = 10']
	time_taken_K10 = ['K = 10']     

	acc_K10, time_K10 = Execute_KNN_TT(10, X_norm, 0.7, distance_func)
	results_K10.extend([round(acc_K10,2)])
	time_taken_K10.extend([round(time_K10,2)])
	acc_K10, time_K10 = Execute_KNN_TT(10, X_norm, 0.6, distance_func)
	results_K10.extend([round(acc_K10,2)])
	time_taken_K10.extend([round(time_K10,2)])
	acc_K10, time_K10 = Execute_KNN_TT(10, X_norm, 0.5, distance_func)
	results_K10.extend([round(acc_K10,2)])
	time_taken_K10.extend([round(time_K10,2)])

	acc_K10, time_K10 = Execute_KNN_CV(10, X_norm, 5, distance_func)
	results_K10.extend([round(acc_K10,2)])
	time_taken_K10.extend([round(time_K10,2)])
	acc_K10, time_K10 = Execute_KNN_CV(10, X_norm, 10, distance_func)
	results_K10.extend([round(acc_K10,2)])
	time_taken_K10.extend([round(time_K10,2)])
	acc_K10, time_K10 = Execute_KNN_CV(10, X_norm, 15, distance_func)
	results_K10.extend([round(acc_K10,2)])
	time_taken_K10.extend([round(time_K10,2)])

	#K = 15
	results_K15 = ['K = 15']
	time_taken_K15 = ['K = 15']

	acc_K15, time_K15 = Execute_KNN_TT(15, X_norm, 0.7, distance_func)
	results_K15.extend([round(acc_K15,2)])
	time_taken_K15.extend([round(time_K15,2)])
	acc_K15, time_K15 = Execute_KNN_TT(15, X_norm, 0.6, distance_func)
	results_K15.extend([round(acc_K15,2)])
	time_taken_K15.extend([round(time_K15,2)])
	acc_K15, time_K15 = Execute_KNN_TT(15, X_norm, 0.5, distance_func)
	results_K15.extend([round(acc_K15,2)])
	time_taken_K15.extend([round(time_K15,2)])

	acc_K15, time_K15 = Execute_KNN_CV(15, X_norm, 5, distance_func)
	results_K15.extend([round(acc_K15,2)])
	time_taken_K15.extend([round(time_K15,2)])
	acc_K15, time_K15 = Execute_KNN_CV(15, X_norm, 10, distance_func)
	results_K15.extend([round(acc_K15,2)])
	time_taken_K15.extend([round(time_K15,2)])
	acc_K15, time_K15 = Execute_KNN_CV(15, X_norm, 15, distance_func)
	results_K15.extend([round(acc_K15,2)])
	time_taken_K15.extend([round(time_K15,2)])

	#K = 20
	results_K20 = ['K = 20']
	time_taken_K20 = ['K = 20']     

	acc_K20, time_K20 = Execute_KNN_TT(20, X_norm, 0.7, distance_func)
	results_K20.extend([round(acc_K20,2)])
	time_taken_K20.extend([round(time_K20,2)])
	acc_K20, time_K20 = Execute_KNN_TT(20, X_norm, 0.6, distance_func)
	results_K20.extend([round(acc_K20,2)])
	time_taken_K20.extend([round(time_K20,2)])
	acc_K20, time_K20 = Execute_KNN_TT(20, X_norm, 0.5, distance_func)
	results_K20.extend([round(acc_K20,2)])
	time_taken_K20.extend([round(time_K20,2)])

	acc_K20, time_K20 = Execute_KNN_CV(20, X_norm, 5, distance_func)
	results_K20.extend([round(acc_K20,2)])
	time_taken_K20.extend([round(time_K20,2)])
	acc_K20, time_K20 = Execute_KNN_CV(20, X_norm, 10, distance_func)
	results_K20.extend([round(acc_K20,2)])
	time_taken_K20.extend([round(time_K20,2)])
	acc_K20, time_K20 = Execute_KNN_CV(20, X_norm, 15, distance_func)
	results_K20.extend([round(acc_K20,2)])
	time_taken_K20.extend([round(time_K20,2)])

    	table_results = tabulate.tabulate([results_K1, results_K5, results_K10, results_K15, results_K20], headers = header_results, tablefmt = 'psql', numalign = 'center')
    	table_time = tabulate.tabulate([time_taken_K1, time_taken_K5, time_taken_K10, time_taken_K15, time_taken_K20,], headers = header_time, tablefmt = 'psql', numalign = 'center')

    	#print(table)
    	f = open('results.txt', 'a')
    	f.write(distance_header)
    	f.write('\n')
    	f.write(table_results)
    	f.write('\n\n')
    	f.write(table_time)
    	f.write('\n\n')
    	f.close()

if __name__=='__main__':

    data = sys.argv[1]
    X = loadData(data)

    print X
    X_norm = dataNorm(X)
    # print testNorm([X_norm])

    #Record_Results(X_norm)

    
    



