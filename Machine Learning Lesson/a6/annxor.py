from scipy.special import expit as sigmoid
from matplotlib.ticker import FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np 
import sys

def loadData(filename):

    X = []
    count = 0
    text_file = open(filename, 'r')
    lines = text_file.readlines()

    for line in lines:

	X.append([])
	words = line.split()

	for word in words:

	    X[count].append(float(word))

	count += 1

    X_ones = np.ones((len(X), 1))
    X = np.hstack((X_ones, X))

    return np.asarray(X)

def paraIni():

    #wh=np.array([[0.1859,-0.7706,0.6257],[-0.7984,0.5607,0.2109]])
    #wo=np.array([[0.1328,0.5951,0.3433]])

    wh = 2 * np.random.random((2, 3)) - 1
    wo = 2 * np.random.random((1, 3)) - 1
		
    return [wh, wo]

def feedforward(X, W):

    X_input = X[:, 0: len(X[0]) - 1]

    Oh = np.tanh(np.dot(W[0], X_input.transpose()))
    n, m = Oh.shape
    Ino_ones = np.ones((1, m))
    Ino = np.vstack((Ino_ones, Oh))
    Oo = sigmoid(np.dot(W[1], Ino))

    return [Oh, Ino, Oo]

def errCompute(X, Oo):

    X_output = X[:, len(X[0]) - 1: len(X[0])]
    cost = np.sum((X_output - Oo.transpose()) ** 2) / (2 * len(X)) 

    return cost

def backpropagate(X, W, intermRslt, alpha):

    X_input = X[:, 0: len(X[0]) - 1]
    X_output = X[:, len(X[0]) - 1: len(X[0])]

    delta_o = np.multiply(np.multiply((X_output.transpose() - intermRslt[2]), intermRslt[2]), (1 - intermRslt[2]))

    wo_updated = W[1] + (alpha * np.dot(delta_o, intermRslt[1].transpose())) / 4.0
    
    dot_product = np.dot(wo_updated[:, 1: len(W[0][0])].transpose(), delta_o)

    element_multi = np.multiply((1 - intermRslt[0]), intermRslt[0])

    delta_h = np.multiply(dot_product, element_multi)
    
    wh_updated = W[0] + (alpha * np.dot(delta_h, X_input)) / 4.0

    return [wh_updated, wo_updated]

def FFMain(X, numIteration, alpha):

    W = paraIni()
  
    #number of features
    n = X.shape[1]
  
    #error
    errHistory = np.zeros((numIteration, 1))
  
    for i in range(numIteration):

	#feedforward
	intermRslt = feedforward(X, W)
	#Cost function
	errHistory[i,0] = errCompute(X, intermRslt[2])
	#backpropagate
	W = backpropagate(X, W, intermRslt, alpha)

    Yhat = np.around(intermRslt[2]) 

    return [errHistory, intermRslt[2], W]

def plotResults(X):

    alpha = [0.01, 0.5]
    numIteration = [100, 1000, 5000, 10000]
    count_num = count_alpha = 0
    fig = plt.figure('Error VS Interation')
    fig.subplots_adjust(hspace = 0.6, wspace = 0.4)
    fig.suptitle('Error VS Interation')
    axs = fig.subplots(len(numIteration), len(alpha))

    for value in alpha:

	for num in numIteration:

	    R = FFMain(X, num, value)
	    axs[count_num, count_alpha].set_title('Alpha: ' + str(value) + ' Iteration: ' + str(num))
	    axs[count_num, count_alpha].set_xlabel('Iteration')
	    axs[count_num, count_alpha].set_ylabel('Error')
   	    axs[count_num, count_alpha].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
	    axs[count_num, count_alpha].plot(R[0])
            count_num += 1
        
        count_num = 0
	count_alpha += 1

    #plt.show(block = False)
    plt.show()

def predict(test, W):

    Oh = np.tanh(np.dot(W[0], test.transpose()))
    Oh = np.expand_dims(Oh, axis = 1)
    Ino_ones = np.ones((1, 1))
    Ino = np.vstack((Ino_ones, Oh))
    Oo = sigmoid(np.dot(W[1], Ino))

    return Oo


if __name__=='__main__':

    X = loadData(sys.argv[1])
    #plotResults(X)

    #W = paraIni()
    #intermRslt = feedforward(X, W)
    #cost = errCompute(X, intermRslt[2])
    #W_updated = backpropagate(X, W, intermRslt, 0.5)
    R = FFMain(X, 10000, 0.5)
    
    test = np.array([1, 0.5, 1.0])
    result = predict(test, R[2])
    
    print(result[0][0])
    #print(R[1])

    plt.plot(R[0])
    plt.show()





