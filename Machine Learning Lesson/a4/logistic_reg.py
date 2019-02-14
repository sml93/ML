from scipy.special import expit as sigmoid

import matplotlib.pyplot as plt
import numpy as np

import sys

def loadData(filename):

    #load data from filename into X
    X = []
    count = 0 
    text_file = open(filename, 'r')
    lines = text_file.readlines()

    for line in lines:
	
	X.append([])
	words = line.split(',')	

	for word in words:

	    X[count].append(float(word))
	
	count += 1

    return np.asarray(X)

def dataNorm(X):
  
    X_norm = []
    X_output = [] 
    X_ones = np.ones((len(X), 1))
    X_output = np.reshape(X[:, 4], (len(X), 1))
    X_min = np.min(X[:, 0:4], axis = 0)
    X_max = np.max(X[:, 0:4], axis = 0)
    X_norm = (X[:, 0:4] - X_min) / (X_max - X_min)
    X_norm = np.hstack((X_norm, X_output))
    X_norm = np.hstack((X_ones, X_norm))

    return np.asarray(X_norm)

def testNorm(X_norm):

    xMerged = np.copy(X_norm[0])

    # Merge datasets
    for i in range(len(X_norm)-1):

        xMerged = np.concatenate((xMerged,X_norm[i+1]))

    print np.mean(xMerged,axis=0)
    print np.sum(xMerged,axis=0)

def errCompute(X_norm, theta):

    actual_response = X_norm[:, 5:6]
    predictor_set = X_norm[:, 0:5]
    prediction_set = sigmoid(np.dot(predictor_set, theta))

    cost = (-1) * np.sum(actual_response * np.log(prediction_set) + (1 - actual_response) * np.log(1 - prediction_set)) / len(X_norm)

    return cost

def stochasticGD(X_norm, theta, alpha, num_iters):

    actual_response = X_norm[:, 5:6]
    predictor_set = X_norm[:, 0:5]
    sample_size = len(X_norm)
	
    for iteration in range(0, num_iters):
	
	# Use random index for random samples 
	rand_ind = np.random.randint(0, len(X_norm)) 

	# Or use iteration % sample_size        

	prediction_set = sigmoid(np.dot(predictor_set, theta))
	loss = (prediction_set[rand_ind] - actual_response[rand_ind]).reshape(1, 1)
	gradient = np.dot(loss, predictor_set[rand_ind].reshape(1, 5)).transpose()
	theta = theta - alpha * gradient

    return theta

def stochasticGD_V(X_norm, theta, alpha, num_iters, i):

    iter_list = []
    err_list = []
    actual_response = X_norm[:, 5:6]
    predictor_set = X_norm[:, 0:5]
    sample_size = len(X_norm)
	
    for iteration in range(0, num_iters):
	
	# Use random index for random samples 
	rand_ind = np.random.randint(0, len(X_norm)) 

	# Or use iteration % sample_size        

	prediction_set = sigmoid(np.dot(predictor_set, theta))
	loss = (prediction_set[rand_ind] - actual_response[rand_ind]).reshape(1, 1)
	gradient = np.dot(loss, predictor_set[rand_ind].reshape(1, 5)).transpose()
	theta = theta - alpha * gradient
	error = errCompute(X_norm ,theta)
	iter_list.append(iteration)
	err_list.append(error)
    
    plt.figure('X_5Set_'+str(i))
    plt.title('Iteration vs Error')
    plt.plot(iter_list, err_list)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()

    return theta

def getAccuracy(prediction_set, actual_response):

    correct = 0
    predict = 0

    for row in range(0, len(prediction_set)):

	if prediction_set[row] >= 0 and prediction_set[row] >= 0.5:
	   
	    predict = 1

	else:

	    predict = 0

	if predict == actual_response[row]:

	    correct += 1

    return (float(correct) / float(len(actual_response))) * 100.0

def split5Set(num_of_set, X_norm):	
    
    np.random.shuffle(X_norm)

    split = int(len(X_norm) / num_of_set)
    X_split_1, X_split_2, X_split_3, X_split_4, X_split_5 = np.split(X_norm, [split, int(split * 2), int(split * 3), int(split * 4)])
    
    return X_split_1, X_split_2, X_split_3, X_split_4, X_split_5

def splitTT(percent, X_norm):	
    
    np.random.shuffle(X_norm)
    split = int(len(X_norm) * percent)
    X_split = np.split(X_norm, [split])
    
    return X_split

def Execute_SGD_TT(X_norm, alpha, num_iters):

    X_train, X_test = splitTT(0.6, X_norm)
    theta = np.zeros((X_norm.shape[1]-1, 1))
    theta = stochasticGD(X_train, theta, alpha, num_iters)
    prediction_set = sigmoid(np.dot(X_test[:, 0:5], theta))
    acc = getAccuracy(prediction_set, X_test[:, 5:6])
    error = errCompute(X_test, theta)

    return acc, error

def findOptimalAlpha(X_norm, num_iters):

    acc_history = []
    alpha_history = []
    optimal_alpha = 0
    highest_acc = 0

    for alpha in np.arange(0.0, 1.5, 0.001):

	acc, error = Execute_SGD_TT(X_norm, alpha, num_iters)
	acc_history.append(acc)
	alpha_history.append(alpha)

	if (acc > highest_acc):

	   highest_acc = acc
	   optimal_alpha = alpha
	    
	   print('Current Optimal Alpha is %s, Current highest Accuracy is %s' %(optimal_alpha, highest_acc))

    plt.title('Optimal Alpha value is ' + str(optimal_alpha) + ', Current Alpha value is: ' + str(alpha))
    plt.plot(alpha_history, acc_history)
    plt.xlabel('Alpha Value')
    plt.ylabel('RMSE')
    plt.pause(1)
    plt.show()

    return optimal_alpha

def Execute_5Sets(X_5Set, optimal_alpha, num_iters):

    for i in np.arange(5):

	X_train, X_test = splitTT(0.6, X_5Set[i])
	theta = np.zeros((X_norm.shape[1]-1, 1))
	theta = stochasticGD_V(X_train, theta, optimal_alpha, num_iters, i)
	prediction_set = sigmoid(np.dot(X_test[:, 0:5], theta))
	acc = getAccuracy(prediction_set, X_test[:, 5:6])
	error = errCompute(X_test, theta)

	print('Set: X_5Set[%s], Accuracy is: %s, Error is %s, Theta is %s' %(i, acc, error, theta))

if __name__=='__main__':

    X = loadData(sys.argv[1])
    #X_actual_response = loadData(sys.argv[2])

    X_norm = dataNorm(X)
    #testNorm([X_norm])
    #error = errCompute(X_norm, np.zeros((X_norm.shape[1]-1,1)))

    optimal_alpha = findOptimalAlpha(X_norm, 1500)
    X_5Set = split5Set(5, X_norm)
    Execute_5Sets(X_5Set, optimal_alpha, 10000)

  




