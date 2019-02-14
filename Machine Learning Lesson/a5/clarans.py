import numpy as np
from random import randint

sample_set = range(0,100)
medoid = []
k = 10

def DistanceBetween(a, b):
	return abs(a-b)

def AllocateContainer(sample_set, medoid, k):
	container = []
	# Create K Containers 
	for idx in range(k):
		container.append([])

	# Allocate Objects into Nearest Sets
	for sample in sample_set:
		distance    = 999
		closest_set = 0
		# Find Nearest Container
		for idx in range(len(medoid)):
			if DistanceBetween(sample, medoid[idx]) < distance:
				distance = DistanceBetween(sample, medoid[idx])
				closest_set = idx

		container[closest_set].append(sample)

	return container

def CalculateContainerScore(container, medoid):
	score = 0
	for m in range(len(medoid)):
		for point in container[m]:
			score += DistanceBetween(point, medoid[m])

	return score

def Process(sample_set, medoid, k):
	# One Single Process include Generating 1 Container
	container = AllocateContainer(sample_set, medoid, k)
	# And calculating score to evaluate that container
	score = CalculateContainerScore(container, medoid)
	# Print for Logging
	#print "Set :", container, "Score :",score
	# Return Score
	return score

## Demostrate Brute Force 
def RecursivelyFind(sample_set, medoid, k, max_search_depth):
	score = Process(sample_set, medoid, k)
	if max_search_depth is 0:
		return [score, sample_set]
	else:
	   updated_depth = max_search_depth - 1
	# Jumble Up a Random Medoid
	# Choose Random Medoid to find
	idx = randint(0, k - 1)
	# Choose a Random Medoid from Sample Set
	new_medoid = medoid[idx]
	while new_medoid in medoid:
		new_medoid = sample_set[randint(0, len(sample_set) - 1)]
	# Perform Swap
	medoid[idx] = new_medoid
	# Check if shuffled one is better
	new_result = RecursivelyFind(sample_set, medoid, k, updated_depth)
	# Compare Score
	if score > new_result[0]:
		#print "Best Score :", new_result[0], "Medoid :", new_result[1]
		return new_result
	else:
		#print "Best Score :", score, "Container :", sample_set
		return [score, medoid]

## Run Script
for i in range(10):
	medoid.append(randint(0, 100))
avg_score = 0	
for i in range(100):
	avg_score += RecursivelyFind(sample_set, medoid, k, 100)[0]

avg_score = float(avg_score) / 100.0

print "Average Container Score", avg_score



