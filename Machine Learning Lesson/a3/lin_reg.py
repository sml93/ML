import numpy as np
import matplotlib.pyplot as plt

import sys

def loadData(filename):

    #load data from filename into X
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

    return np.asarray(X)

def dataNorm(X):
  
    X_norm = []
    X_output = [] 
    X_ones = np.ones((len(X), 1))
    X_output = np.reshape(X[:, 13], (len(X), 1))
    X_min = np.min(X[:, 0:13], axis = 0)
    X_max = np.max(X[:, 0:13], axis = 0)
    X_norm = (X[:, 0:13] - X_min) / (X_max - X_min)
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

def scatterPlot(X):

    fig = plt.figure('Feature - Price Plot')
    fig.suptitle('Feature - Price Plot')
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    fig.add_subplot(3, 2, 1)
    plt.scatter(X[:, 12], X[:, 13], alpha = 0.5)
    plt.xlabel('LSTAT')
    plt.ylabel('MEDV (in $1000)')
    plt.title('LSTAT against MEDV (in $1000)')

    fig.add_subplot(3, 2, 2)
    plt.scatter(X[:, 7], X[:, 13], alpha = 0.5)
    plt.xlabel('DIS (Weighted distances to five Boston employment centers)')
    plt.ylabel('MEDV (in $1000)')
    plt.title('DIS against MEDV (in $1000)')

    fig.add_subplot(3, 2, 3)
    plt.scatter(X[:, 6], X[:, 13], alpha = 0.5)
    plt.xlabel('AGE (Proportion of owner-occupied units built prior to 1940)')
    plt.ylabel('MEDV (in $1000)')
    plt.title('AGE against MEDV (in $1000)')    

    fig.add_subplot(3, 2, 4)
    plt.scatter(X[:, 5], X[:, 13], alpha = 0.5)
    plt.xlabel('RM (Average number of rooms per dwelling)')
    plt.ylabel('MEDV (in $1000)')
    plt.title('RM against MEDV (in $1000)')    

    fig.add_subplot(3, 2, 5)
    plt.scatter(X[:, 0], X[:, 13], alpha = 0.5)
    plt.xlabel('CRIM (Per capita crime rate by town)')
    plt.ylabel('MEDV (in $1000)')
    plt.title('CRIM against MEDV (in $1000)')

    plt.show()

def errCompute(X_norm, theta):

    actual_response = X_norm[:, 14:15]
    predictor_set = X_norm[:, 0:14]

    prediction_set = np.dot(predictor_set, theta)
    cost = np.sum(pow(actual_response - prediction_set, 2)) / (2 * len(X_norm))

    return cost

def gradientDescent(data, theta, alpha, num_iters):

    actual_response = data[:, 14:15]
    predictor_set = data[:, 0:14]

    for iteration in range(0, num_iters):
	
        prediction_set = np.dot(predictor_set, theta)
	loss = (prediction_set - actual_response).transpose()
	print(loss.shape)
	gradient = (np.dot(loss, predictor_set) / len(data)).transpose()      
	theta = theta - alpha * gradient
        
    return theta
    
def splitCV(k, X_norm):

    np.random.shuffle(X_norm)
    X_split = np.array_split(X_norm, k)
    
    return X_split

def Execute_GD_CV_Plot(X_norm, alpha, num_iters):

    fold_data = [5, 10, 15]
    fig = plt.figure('Experimental Results')
    fig.subplots_adjust(hspace = 0.6, wspace = 0.4)
    axs = fig.subplots(3, 1)

    for k in fold_data:

	X_splitCV = splitCV(k, X_norm)
	total_error = 0        
	error = 0

    	for j in range(k):

	    X_test = np.copy(X_splitCV[j])
	    X_train = np.array([])

	    for l in range(k):

	        if (j == l):
		
		    continue
		
                if (X_train.size == 0):

		    X_train = X_splitCV[l]

	        else:
		
		    X_train = np.concatenate((X_train, X_splitCV[l]))

	    theta = np.zeros((14, 1))
	    theta = gradientDescent(X_train, theta, alpha, num_iters)
	    predict_price = np.dot(X_train[:, 0:14], theta)
	    rmse = np.sqrt(np.mean(pow((predict_price - X_train[:, 14-15]), 2)))
	    train_set = X_train[:, 14-15].reshape(len(X_train), 1)
	    
	    #give different ans why?
            #rmse = np.sqrt( np.mean((predict_price - X_train[:, 14:15]) ** 2))

  	    error += rmse

	if (k == 5):

	    max_lim = np.max(predict_price, axis = 0)
	    axs[0].set_xlim([0, max_lim])
    	    axs[0].set_ylim([0, max_lim])
    	    axs[0].scatter(predict_price, train_set, alpha = 0.5)
    	    axs[0].plot([0, max_lim], [0, max_lim], 'k-', color = 'r')
	    axs[0].set_xlabel('Predicted Prices')
    	    axs[0].set_ylabel('Actual Prices')
    	    axs[0].set_title('Predicted vs Actual Prices (K = ' + str(k) + ')')
    
    	if (k == 10):

	    max_lim = np.max(predict_price, axis = 0)
	    axs[1].set_xlim([0, max_lim])
    	    axs[1].set_ylim([0, max_lim])
    	    axs[1].scatter(predict_price, train_set, alpha = 0.5)
    	    axs[1].plot([0, max_lim], [0, max_lim], 'k-', color = 'r')
	    axs[1].set_xlabel('Predicted Prices')
    	    axs[1].set_ylabel('Actual Prices')
    	    axs[1].set_title('Predicted vs Actual Prices (K = ' + str(k) + ')')

    	if (k == 15):

	    max_lim = np.max(predict_price, axis = 0)
	    axs[2].set_xlim([0, max_lim])
    	    axs[2].set_ylim([0, max_lim])
            axs[2].scatter(predict_price, train_set, alpha = 0.5)
            axs[2].plot([0, max_lim], [0, max_lim], 'k-', color = 'r')
            axs[2].set_xlabel('Predicted Prices')
            axs[2].set_ylabel('Actual Prices')
            axs[2].set_title('Predicted vs Actual Prices (K = ' + str(k) + ')')
	
	total_error = error / k

	print('Cross Validation Folds: %s, result in RMSE: %s' %(k, total_error))

    plt.show()

def Execute_GD_CV(X_norm, alpha, num_iters):

    fold_data = [5, 10, 15]

    for k in fold_data:

	X_splitCV = splitCV(k, X_norm)
	total_error = 0        
	error = 0

    	for j in range(k):

	    X_test = np.copy(X_splitCV[j])
	    X_train = np.array([])

	    for l in range(k):

	        if (j == l):
		
		    continue
		
                if (X_train.size == 0):

		    X_train = X_splitCV[l]

	        else:
		
		    X_train = np.concatenate((X_train, X_splitCV[l]))

	    theta = np.zeros((14, 1))
	    theta = gradientDescent(X_train, theta, alpha, num_iters)

	    predict_price = np.dot(X_train[:, 0:14], theta)
	    #rmse = np.sqrt(np.mean(pow((predict_price - X_train[:, 14-15]), 2)))
	    train_set = X_train[:, 14-15].reshape(len(X_train), 1)
	    
	    #give different ans why?
            rmse = np.sqrt(np.mean((predict_price - X_train[:, 14:15]) ** 2))
	
  	    error += rmse

	total_error = error / k

	return total_error
	#print('Cross Validation Folds: %s, result in RMSE: %s' %(k, total_error))

def findOptimalAlpha(X_norm, num_iters):

    error_history = []
    alpha_history = []
    optimal_alpha = 0
    lowest_rmse = 999

    for alpha in np.arange(0.0, 0.5, 0.001):

	rmse = Execute_GD_CV(X_norm, alpha, num_iters)
	error_history.append(rmse)
	alpha_history.append(alpha)

	if (rmse < lowest_rmse):

	   lowest_rmse = rmse
	   optimal_alpha = alpha
	    
	   print('Current Optimal Alpha is %s, Current lowest RMSE is %s' %(optimal_alpha, lowest_rmse))

    plt.title('Optimal Alpha, Current Alpha value is: '+ str(alpha))
    plt.plot(alpha_history, error_history)
    plt.xlabel('Alpha Value')
    plt.ylabel('RMSE')
    plt.pause(1)
    plt.show()

def VisualizeGradientDescent(data, alpha, num_iters):

    theta = np.zeros((14,1))
    predictor_set = data[:, 0:14]
    expected_set  = data[:, 14:15]
    alpha_constant = alpha / len(data)

    for iteration in range(0, num_iters):
        
	prediction_set = np.dot(predictor_set, theta)
	error = (prediction_set - expected_set).transpose()
	gradient = np.dot(error, predictor_set) * alpha_constant
	theta = theta - np.sum(gradient, axis = 0).reshape(14,1)
	plt.clf()
    	plt.plot(prediction_set, expected_set, 'o')
    	plt.title("Predicted vs Expected Value at Iteration = " + `iteration`)
    	plt.xlabel("Predicted Value")
    	plt.ylabel("Expected Value")
    	plt.plot([0, 50], [0, 50], 'k-', lw=1)
    	plt.xlim(0,50)
   	plt.ylim(0,50)
    	plt.pause(0.05)
  # Magic!
    plt.show()

if __name__ == '__main__':

    X = loadData(sys.argv[1])
    X_norm = dataNorm(X)
    Execute_GD_CV(X_norm, 0.417, 1000)
    #findOptimalAlpha(X_norm, 1500)
    #VisualizeGradientDescent(X_norm, 0.5, 1500)

    #scatterPlot(X)
    #cost = errCompute(X_norm, np.zeros((X_norm.shape[1]-1,1)))
    #testNorm([X_norm])

    










