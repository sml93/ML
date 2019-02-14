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

if __name__=='__main__':

    X = loadData(sys.argv[1])
    print(len(X[0]))
