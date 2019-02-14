#!/usr/bin/python
##
# \file   kmeans.py
# \author Keechin Goh, g.keechin, 390005015
# \date   18 July 2018
# \brief  This file contains functions to run logistic regression
#
# \usage python kmeans.py <path-to-datafile> <k-clusters>
# \sample python kmeans.py 2010825.txt 10
#
# Copyright (C) 2018 DigiPen Institute of Technology.
# Reproduction or disclosure of this file or its contents without the prior
# written consent of DigiPen Institute of Technology is prohibited.
import sys
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Global Variable for K (Default 5)
K_CLUSTERS = 5

# Error Compute to Monitor Convergence
def errCompute(data, means):
  # Summation Initialization
  error = 0
  # Per Data Check
  for row in data:
    error += math.sqrt((row[0] - means[row[2]][0])**2 + (row[1] - means[row[2]][1])**2)

  return error / len(data)

# Grouping Function
def Group(data, means):
  # Iterate Through All Objects
  for entry in data:
    # Variable for Finding Closest Group
    group_counter = 0
    closest_group = 0
    closest_distance  = float("inf")
    # Iterate Through Each Group
    for group in means:
      # Calculate Distance from Centroid
      distance = (entry[0] - group[0])**2 + (entry[1] - group[1])**2
      # Find Lowest Distance and Group it
      if(distance < closest_distance):
        closest_group = group_counter
        closest_distance = distance
      # Increment Group
      group_counter += 1

    # Set Grouping Index
    entry[2] = closest_group
  
  return data

# Calculate New Means
def calcMeans(data, means):
  new_mean   = np.zeros((K_CLUSTERS, 2))
  group_size = np.zeros(K_CLUSTERS)

  # Iterate Through Data
  for entry in data:
    # Add Data to Group Summation
    new_mean[entry[2]] += entry[0:2]
    # Increase Group Membership Counter
    group_size[entry[2]] += 1

  # Obtain Average (Mean)
  return new_mean / group_size[:,None]

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
    # Append Additional Column for GroupID
    X[counter].append(int(0))
    counter += 1

  return np.asarray(X)

def VisualizeData(data):
  plt.title("Lightning Strikes Locations Overview")
  plt.plot(data[:,0], data[:,1], 'x')
  plt.savefig("visualize.png", dpi=150)

def VisualizeScatter(data):
  plt.title("Clustering Overview")
  # Define Colours
  colors = [plt.cm.gist_ncar(i) for i in np.linspace(0, 0.9,K_CLUSTERS)]   
  color_array = []
  for entry in data:
    color_array.append(colors[int(entry[2])])
  
  plt.scatter(data[:,0], data[:,1], alpha = 0.7, c = color_array)
  plt.savefig("scatter.png", dpi=150)

      
def PrintTime():
  return "[" + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "]"

def main():
  print "Using Data Source : ", sys.argv[1], " - Splitting", sys.argv[2], "Clusters"
  X_data = loadData(sys.argv[1])
  
  # Set Cluster Value
  global K_CLUSTERS
  K_CLUSTERS = int(sys.argv[2])

  # Create Initial Mean
  M = np.copy(X_data[0:K_CLUSTERS, 0:X_data.shape[1]-1])

  error_history = []
  iteration_history = []

  # Iterate Through Range
  for i in range(100):
    K_CLUSTERS = 1 + i
    X_data = Group(X_data, M)
    M = calcMeans(X_data, M)
    error_history.append(errCompute(X_data, M))
    iteration_history.append(i)
    plt.clf()
    plt.title("Error Change K")
    plt.ylabel("Error");
    plt.xlabel("K Clusters")
    plt.plot(iteration_history, error_history)
  
  plt.show()
  print min(error_history)
  VisualizeScatter(X_data)



if __name__ == "__main__":
    main() 
