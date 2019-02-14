#!/usr/bin/python
##
# \file   lin_reg.py
# \author Keechin Goh, g.keechin, 390005015
# \date   24 June 2018
# \brief  This file contains functions to run linear regression analysis
#
# \usage python lin_reg.py <path-to-datafile>
#
# Copyright (C) 2018 DigiPen Institute of Technology.
# Reproduction or disclosure of this file or its contents without the prior
# written consent of DigiPen Institute of Technology is prohibited.
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Error Compute to Monitor Convergence
def errCompute(data, theta):
  # Split Data and Output
  predictor_set = data[:, 0:14]
  expected_set  = data[:, 14:15]

  # Perform Summation
  prediction_set = np.dot(predictor_set, theta)
  # Calculate Error
  error = np.sum((expected_set - prediction_set) ** 2)

  return error / (2 * len(data))

# Gradient Descent
def gradientDescent(data, theta, alpha, num_iter):
  # Split Data and Output
  predictor_set = data[:, 0:14]
  expected_set  = data[:, 14:15]

  # Cache Alpha Constant (a / N)
  alpha_constant = alpha / len(data)

  # For Each Proposed Iteration
  for iteration in range(0, num_iter):
    # Obtain Predicted Values 
    prediction_set = np.dot(predictor_set, theta)

    # Calculate (Yi^ - Yi) * Xi
    error = (prediction_set - expected_set).transpose()

    # Gradient Value
    gradient = np.dot(error, predictor_set) * alpha_constant

    # Update Original Theta
    theta = theta - np.sum(gradient, axis = 0).reshape(14,1)

  return theta

# Root Mean Squared Error
def rmse(testY, stdY):
  return np.sqrt( np.mean((testY - stdY) ** 2) )

# Cross Validation K Split
def splitCV(data, folds):
  shuffled_data = data;
  np.random.shuffle(shuffled_data)
  return np.array_split(shuffled_data, folds)

# Normalize Data Attributes
def dataNorm(data):
  col_max_values = data.max(0)
  col_min_values = data.min(0)
  X_norm = np.empty((0, 14), float)

  # Zip and Calculated Per-Loop
  for row in data:
    X_norm = np.vstack([X_norm, [(x - min) / (max - min) for x, min, max in zip(row, col_min_values, col_max_values)]])
  
  # Replace Last Column
  X_norm[:,13] = data[:,13]

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
    values = line.split()
    # Append Value per Row
    for value in values:
      X[counter].append(float(value))
    counter += 1

  return np.asarray(X)

# Scatter Plot for Feature-Price Analysis
def ScatterPlot(data, col):
  # Label / Description of Chart
  label = ['CRIM (per capita)', 'ZN', 'INDUS','CHAS', 'NOX', 'RM (avg. room per dwelling)', 'AGE (proportion of owner-occupied units built prior to 1940)',
   'DIS (weighted distances to five Boston employment centres)', 'RAD', 'TAX', 'PTRATIO',
   'B', 'LSTAT (% lower status of the population)']
  label_sim = ['CRIM', 'ZN', 'INDUS','CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

  plt.plot(data[:,col], data[:,13], 'o')
  plt.xlabel(label[col])
  plt.ylabel('MEDV (in \'$1000)')
  plt.title(label_sim[col] + ' against ' + 'MEDV (in \'$1000)')
  plt.savefig('plot-' + `col` + '.png', dpi = 150)
  plt.clf()

# Split-CV Test for Alpha
def RunTest(data, alpha, iterations_to_run):
  # Announce
  print PrintTime() + " Gradient Descent Test [Alpha = " + str(alpha) + "] over " + str(iterations_to_run) + " iterations."
  # Folds
  k_folds = [5, 10, 15]

  # Execute Test per Fold
  for fold in k_folds:
    # Split Data 
    split_data = splitCV(data, fold)
    # Total Error
    total_error = 0

    # Run for Each Fold Iteration
    for iteration in range(fold):
      # Prepare Containers
      training_set = np.empty((0, 15), float)
      testing_set  = split_data[iteration]
      # Extract Training Data Except Iteration Value
      for train in [x for i, x in enumerate(split_data) if i != iteration]:
        training_set = np.concatenate((training_set, train))

      # Begin with Zeroed-Theta
      training_theta = np.zeros((14,1))
      # Run Gradient Descent with Training Set
      training_theta = gradientDescent(training_set, training_theta, alpha, iterations_to_run)
      # Obtain Final Prediction
      predicted_prices = np.dot(training_set[:, 0:14], training_theta)
      error = rmse(predicted_prices, training_set[:, 14:15])
      # Accumulate Error
      total_error += error
    
    # Announce
    print PrintTime() + " Average RMSE using " + str(fold) + " folds :\t" + str(total_error / fold)

  print "---------------------------------------------------------------------------------"

  return total_error / fold
      
def PrintTime():
  return "[" + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "]"

def main():
  print "Using Data Source : ", sys.argv[1]
  X_data = loadData(sys.argv[1])
  X_norm = dataNorm(X_data)

  # Actual Test Running
  RunTest(X_norm, 0.417, 2000)

if __name__ == "__main__":
    main() 
