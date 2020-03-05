from math import exp
import numpy as np
import re

######
# NEURON CLASS
# - Basic Neuron to analysis of threshold, linear 
# and sigmoid classification. 
######
class Neuron:

	def __init__(self,nweights,bias=1):
		self.weight = np.ones(nweights)
		self.bias = bias

	def threshold_classify(self,x):
		v = np.dot(x,self.weight) + self.bias
		y = 2 if v + bias >= 0 else 1
		return y

	def linear_classify(self,x):
		v = np.dot(x,self.weight) + self.bias

		if v <= -1/2:
			return 1
		elif -1/2 < v < 1/2:
			return v
		else:
			return 2

	def sigmoid_classify(self,x,alpha=1):
		v = np.dot(x,self.weight) + self.bias
		y = 1/(1 + exp(-alpha*v))
		return y + 1

######
# MAIN CODE
######
# 1. Reading the data
number_of_examples = 0
v1, v2, v3 = [], [], []
with open('Aula2-exec1.csv') as inputfile:
	for line in inputfile:
		if re.match("^\d+.\d+,\d+.\d+,\d+$",line) is not None:
			data = line.split(',')
			v1.append(float(data[0]))
			v2.append(float(data[1]))
			v3.append(int(data[2]))
			number_of_examples += 1

# 2. Initializing the Neuron
neuron = Neuron(nweights=2)

# 3. Analysing the best weight and bias to the proposed problem
print "# Training"
threshold_result = []
linear_result = []
sigmoid_result = []
for bias in np.arange(-5,5,0.5):
	print "->",bias
	for w1 in np.arange(-5,5,0.5):
		for w2 in np.arange(-5,5,0.5):
			# a. initialising the errors
			threshold_error = 0
			linear_error = 0
			sigmoid_error = 0

			# b. testing the weights for each input
			for i in range(number_of_examples):
				x, y = np.array([v1[i],v2[i]]),v3[i]

				# d. defining neuron parameters
				neuron.weight = np.array([w1,w2])
				neuron.bias = bias

				# e. calculating the error
				threshold_error += abs(round(neuron.threshold_classify(x))-y)
				linear_error += abs(round(neuron.linear_classify(x))-y)
				sigmoid_error += abs(round(neuron.sigmoid_classify(x))-y)
			
			threshold_result.append([bias,w1,w2,threshold_error])
			linear_result.append([bias,w1,w2,linear_error])
			sigmoid_result.append([bias,w1,w2,sigmoid_error])

# 4. Getting the best bias and weights to the problem for each activation func
min_error, t_best_config = 99999999, None
for config in threshold_result:
	if config[3] < min_error:
		min_error = config[3]
		best_config = config
print "Threshold:",best_config

min_error, l_best_config = 99999999, None
for config in linear_result:
	if config[3] < min_error:
		min_error = config[3]
		best_config = config
print "Linear:",best_config

min_error, s_best_config = 99999999, None
for config in sigmoid_result:
	if config[3] < min_error:
		min_error = config[3]
		best_config = config
print "Sigmoid:",best_config

# 4. Plotting the result
import matplotlib.pyplot as plt
plt.plot(v1,v2)