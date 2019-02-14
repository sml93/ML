# the structure of neural network: 
#    input layer with 2 inputs
#    1 hidden layer with 2 units, tanh()
#    output layer with 1 unit, sigmoid()

import numpy as np
import scipy
from   scipy.special import expit

def paraIni():
  #code for fixed network and initial values
  
  # parameters for hidden layer, 2 by 3
  wh=np.array([[0.1859,-0.7706,0.6257],[-0.7984,0.5607,0.2109]])
  
  # parameter for output layer 1 by 3
  wo=np.array([[0.1328,0.5951,0.3433]])

  return [wh,wo]
  
def feedforward(X,paras):

  return [oh,ino,oo]
  
def errCompute(Y,Yhat):

  return J

def backpropagate(X,paras,intermRslt,alpha):

  return [wh,wo]
  
def FFMain(filename,numIteration, alpha):
  #data load
  X = loadData(filename)
  #
  W = paraIni()
  
  #number of features
  n = X.shape[1]
  
  #error
  errHistory = np.zeros((numIteration,1))
  
  for i in range(numIteration):
    #feedforward
    intermRslt=feedforward(X,W)
    #Cost function
    errHistory[i,0]=errCompute(X[:,n-1:n],intermRslt[2])
    #backpropagate
    W=backpropagate(X,W,intermRslt,alpha)

  Yhat=np.around(intermRslt[2]) 
  return [errHistory,intermRslt[2],W]
