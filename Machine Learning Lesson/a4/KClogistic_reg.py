#!/usr/bin/python#!/usr/bin/python
##
# \file   logistic_reg.py
# \author Keechin Goh, g.keechin, 390005015
# \date   11 July 2018
# \brief  This file contains functions to run logistic regression
#
# \usage python logistic_reg.py <path-to-datafile>
#
# Copyright (C) 2018 DigiPen Institute of Technology.
# Reproduction or disclosure of this file or its contents without the prior
# written consent of DigiPen Institute of Technology is prohibited.
import sys
import math
import numpy as np
from datetime import datetime
from scipy.special import expit
import matplotlib.pyplot as plt

# Accuracy Function
def computeAccuracy(data, prediction):
  hits = 0
  expected_set  = data[:, 5:6]

  for i in range(0, len(prediction)):
    if(expected_set[i] == prediction[i]):
      hits += 1

  return float(hits) / float(len(expected_set))

# Prediction Function 
def Predict(data, theta):
  # Split Data and Output
  predictor_set = data[:, 0:5]
  expected_set  = data[:, 5:6]

  # Obtain Predicted Values (Sigmoid Function)
  prediction_set = np.dot(predictor_set, theta)
  prediction_set_sigmoid = expit(prediction_set)
  # Create Empty Result (0, 1)
  prediction = np.zeros(shape = expected_set.shape)

  for i in range(0, len(prediction_set)):
    if prediction_set[i] >= 0 and prediction_set_sigmoid[i] >= 0.5:
      prediction[i] = 1
    else:
      prediction[i] = 0

  return prediction

# Error Compute to Monitor Convergence
def errCompute(data, theta):
  # Split Data and Output
  predictor_set = data[:, 0:5]
  expected_set  = data[:, 5:6]

  # Perform Summation
  prediction_set = expit(np.dot(predictor_set, theta))
  # Calculate Error
  error = (-1) * np.sum(expected_set * np.log(prediction_set) + (1 - expected_set) * np.log(1 - prediction_set)) / len(prediction_set)

  return error

# Gradient Descent
def stochasticGD(data, theta, alpha, num_iter):
  # Split Data and Output
  predictor_set = data[:, 0:5]
  expected_set  = data[:, 5:6]

  # Cache Data Length
  sample_size = len(data)

  # For Each Proposed Iteration
  for iteration in range(0, num_iter):
    # Obtain Predicted Values (Sigmoid Function)
    prediction_set = expit(np.dot(predictor_set, theta))

    # Calculate Error Values
    for cell in range(0, 5):
      error = (prediction_set - expected_set)[iteration % sample_size] * predictor_set[iteration % sample_size, cell:cell + 1]
      # Find Corrected Value
      theta[cell] = theta[cell] - alpha * error

  return theta

# Normalize Data Attributes
def dataNorm(data):
  col_max_values = data.max(0)
  col_min_values = data.min(0)
  X_norm = np.empty((0, 5), float)

  # Zip and Calculated Per-Loop
  for row in data:
    X_norm = np.vstack([X_norm, [(x - min) / (max - min) for x, min, max in zip(row, col_min_values, col_max_values)]])
  
  # Replace Last Column
  X_norm[:,4] = data[:,4]

  # Prepend '1' as Counting Variable
  X_norm = np.concatenate((np.full((len(X_norm)), 1)[:, np.newaxis], X_norm), axis=1)

  return X_norm

# Test Normalized Dataset
def testNorm(X_norm):
  xMerged = np.copy(X_norm[0])
  #merge datasets
  for i in range(len(X_norm)-1):
    xMerged = np.concatenate((xMerged,X_norm[i+1]))
  print np.mean(xMerged,axis=0)
  print np.sum(xMerged,axis=0)

# Load Data from Filename into X
def loadData(filename):
  X = []
  counter = 0
  text_file = open(filename, "r")
  lines = text_file.readlines()
    
  for line in lines:
    X.append([])
    values = line.split(",")
    # Append Value per Row
    for value in values:
      X[counter].append(float(value))
    counter += 1

  return np.asarray(X)

# Function to Split Training and Testing Sets
def splitTT(data, percent):
  training_set = data
  np.random.shuffle(training_set)
  split_set = np.split(training_set, [int(len(training_set) * percent)])
  return split_set

# Split-TT 60-40
def RunTest(data, alpha, iterations_to_run):
  # Set Print Properties 
  np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
  # Announce
  print PrintTime() + " Stochastic GD Test [Alpha = " + str(alpha) + "] over " + str(iterations_to_run) + " iterations."

  # Split Test
  split_set = splitTT(data, 0.6)
  # Obtain Learned Theta
  calculated_theta = stochasticGD(split_set[0], np.zeros((data.shape[1] - 1, 1)), alpha, iterations_to_run)
  # Obtain Predictions
  predictions = Predict(split_set[1], calculated_theta)
  print(predictions)
  # Obtain Accuracy 
  accuracy = computeAccuracy(split_set[1], predictions)
  # Print Accuracy
  print "Accuracy : " + str(accuracy)

  return errCompute(data, calculated_theta)

def GraphedAlphaProgression(data):
  accuracy_history = []
  alpha_history = []

  # Testing Variables
  optimal_alpha = 0
  highest_accuracy = 0

  for alpha in np.arange(0.5, 1.5, 0.001):
    result = RunTest(data, alpha, 10000)
    accuracy_history.append(result)
    alpha_history.append(alpha)

    if(result > highest_accuracy):
      highest_accuracy = result
      optimal_alpha = alpha

    plt.clf()
    plt.title("Current Alpha Value: " + str(alpha) + "\n Best Alpha " + str(optimal_alpha) + " [" + str(highest_accuracy) +"] ")
    plt.plot(alpha_history, accuracy_history)
    plt.xlabel("Alpha Value")
    plt.ylabel("Accuracy")
    plt.pause(0.05)
  plt.show()

def IterationVSError(data):
  error_history = []
  iteration_history = []

  for iter in np.arange(0, 10000, 20):
    error = RunTest(data, 1.23, iter)
    error_history.append(error)
    iteration_history.append(iter)

    plt.clf()
    plt.title("Iteration vs Error")
    plt.plot(iteration_history, error_history)
    plt.xlabel("Iterations")
    plt.ylabel("Error (errCompute)")
    plt.pause(0.05)
  plt.show
      
def PrintTime():
  return "[" + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "]"

def main():
  print "Using Data Source : ", sys.argv[1]
  X_data = loadData(sys.argv[1])
  X_norm = dataNorm(X_data)

  # Command to Run a Single Test with Learning Rate 1.23 and 10000 Iterations.
  RunTest(X_norm, 1.23, 10000)

  # Command to Plot (Dynamically) Iteration VS Error 
  #IterationVSError(X_norm)

  # Command to Plot / Find Best Alpha Values 
  # GraphedAlphaProgression(X_norm)

if __name__ == "__main__":
    main() 
