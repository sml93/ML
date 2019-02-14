#!/usr/bin/python
##
# \file   annxor.py
# \author Keechin Goh, g.keechin, 390005015
# \date   3 August 2018
# \brief  This file contains functions to run ANN Functions
#
# \usage python annxor.py <path-to-datafile> <iterations> <alpha>
# \sample python annxor.py 
#
# Copyright (C) 2018 DigiPen Institute of Technology.
# Reproduction or disclosure of this file or its contents without the prior
# written consent of DigiPen Institute of Technology is prohibited.
import sys
import math
import scipy as sc
import numpy as np
from datetime import datetime
from scipy.special import expit
import matplotlib.pyplot as plt


# Load Data from Filename into X
def loadData(filename):
  X = []
  counter = 0
  text_file = open(filename, "r")
  lines = text_file.readlines()

  for line in lines:
    X.append([])
    values = line.split()
    # Append Value per Row
    for value in values:
      X[counter].append(float(value))
    counter += 1

  # Append Row of Ones
  X = np.asarray(X)
  n, m = X.shape
  X_Ones = np.ones((n,1))

  return np.hstack((X_Ones, X))

def FFMain(filename,numIteration, alpha):
  #data load
  X = loadData(filename)
  # Initialize Parameters
  W = paraIni()
  
  # Number of Features
  n = X.shape[1]
  
  # Get Error History
  errHistory = np.zeros((numIteration,1))
  
  for i in range(numIteration):
    # Feed Forward
    intermRslt=feedforward(X,W)
    # Cost function
    errHistory[i,0]=errCompute(X[:,n-1:n],intermRslt[2])
    #backpropagate
    W=backpropagate(X,W,intermRslt,alpha)

  Yhat=np.around(intermRslt[2])
  
  return [errHistory, intermRslt[2], W]

# Initialize Parameters
def paraIni():
  weight_hidden = np.array([[0.1859,-0.7706,0.6257],[-0.7984,0.5607,0.2109]])
  weight_output = np.array([[0.1328,0.5951,0.3433]])
  #weight_hidden = np.random.random_sample((2,3))
  #weight_output = np.random.random_sample((1,3))

  return [weight_hidden, weight_output]

# Feed Forward Algorithm
def feedforward(data, parameters):
  # Extract and Separate Input and Output
  input  = data[:, 0:len(data[0]) - 1]
  output = data[:, len(data[0]) - 1:len(data[0])]

  # Perform Feed Forward
  output_hidden = np.tanh(np.dot(parameters[0], input.transpose()))
  # Obtain Dimension of Output
  n,m = output_hidden.shape
  # Generate Padding Ones
  X_ones = np.ones((m,1))
  # Pad Input_Output
  input_output = np.vstack((X_ones.transpose(), output_hidden))
  # Pass Through Sigmoid Functions
  output = expit(np.dot(parameters[1], input_output))

  return [output_hidden, input_output, output]

def predict(input, parameters):
  # Perform Feed Forward
  output_hidden = np.tanh(np.dot(parameters[0], input.transpose()))
  # Generate Padding Ones
  X_ones = np.ones((2,1))
  # Pad Input_Output
  input_output = np.insert(output_hidden, 0, 1, axis=0)
  # Pass Through Sigmoid Functions
  output = expit(np.dot(parameters[1], input_output))


  return output

# Compute Summation of Errors
def errCompute(expected, result):
  # Setup Summation
  summation = 0
  results = result[0]

  for k in range(len(results)):
    summation += (expected[k] - results[k]) ** 2
  # Obtain Error value
  error = summation / (2 * len(results))

  return error

# Backpropogate Changes and Updates to Weights
def backpropagate(data, parameters, FF_results, alpha):
  # Initializing
  output = data[:, len(data[0])-1:len(data[0])]
  input = data[:, 0:len(data[0])-1]

  oo    = FF_results[2][0]
  ino   = FF_results[1]
  oh    = FF_results[0]
  wh    = parameters[0]
  wo    = parameters[1]

  # Calculate Delta 
  delta = np.multiply(np.multiply((output.transpose() - oo), oo), (1.0-oo))
  wo    = wo + (alpha * np.dot(delta,ino.transpose()))/4.0
  wop   = wo[:, 1:len(wo[0])]
  dot   = np.dot(wop.transpose(), delta)

  deltah = np.multiply(dot, (1.0 - oh*oh))
  wh = wh + alpha * np.dot(deltah, input) / 4.0

  return [wh,wo]

def main():
  print "Using Data Source : ", sys.argv[1]

  # Execute FF Main
  FF_Results = FFMain(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
  #params = np.load('addition.npy') 

  print predict(np.array([1, 0.3, 0.1]), FF_Results[2])

  


if __name__ == "__main__":
    main() 
