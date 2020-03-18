from math import *
from numpy.random import uniform
import numpy as np


class Perceptron:

	def __init__(self,nweight,bias=None):
		self.weight = uniform(-1,1,nweight)
		self.bias = (uniform(-1,1,1))[0] if bias is None else bias

	def classify(self,x):
		y = []
		for sample in x:
			v = np.dot(sample,self.weight) + self.bias
			y.append(1 if v > 0 else 0)
		y = np.array(y) + 1
		return y

	def update(self,x,y,yp,eta):
		for i in range(len(x)):
			self.weight = self.weight + (eta*(x[i,])*(y[i]-yp[i]))
			self.bias = self.bias + (eta*(y[i]-yp[i]))

	def train(self,x,y,eta=1,threshold=0.01):
		result = []
		it, error = 1, 999999
		while(error > threshold):
			# a. classifying
			yp = self.classify(x)

			# b. calculating the error
			error = sum([abs(y[i]-yp[i]) for i in range(len(y))])
			result.append(error)
			#print it,':',error,'( w:',self.weight,', b:',self.bias,')'

			# c. updating the weight
			self.update(x,y,yp,eta)
			it += 1

		return result

	def sigmoid_classify(self,x,alpha=1):
		y = []
		for sample in x:
			v = np.dot(sample,self.weight) + self.bias
			y.append(round(1/(1 + exp(-alpha*v))))
		y = np.array(y) + 1
		return y

	def sigmoid_train(self,x,y,eta=1,threshold=0.01):
		result = []
		it, error = 1, 999999
		while(error > threshold):
			# a. classifying
			yp = self.sigmoid_classify(x)

			# b. calculating the error
			error = sum([abs(y[i]-yp[i]) for i in range(len(y))])
			result.append(error)
			#print it,':',error,'( w:',self.weight,', b:',self.bias,')'

			# c. updating the weight
			self.update(x,y,yp,eta)
			it += 1

		return result

	def euclidean_dist(self,p,q):
		return sqrt(sum([(p[i] - q[i])**2 for i in range(len(self.weight))]))

	def group_distances(self,x,yp):
		dist = np.zeros(2)
		group = [[],[]]

		# a. defining the groups
		for i in range(len(yp)):
			if yp[i] == 1:
				group[0].append(x[i])
			else:
				group[1].append(x[i])

		# b. calculating the distances
		for i in range(len(group[0])):
			for j in range(i,len(group[0])):
				dist[0] += self.euclidean_dist(group[0][i],group[0][j])
		if len(group[0]) > 0:
			dist[0] = dist[0]/len(group[0])
		else:
			dist[0] == 999999

		for i in range(len(group[1])):
			for j in range(i,len(group[1])):
				dist[1] += self.euclidean_dist(group[1][i],group[1][j])
		if len(group[1]) > 0:
			dist[1] = dist[1]/len(group[1])
		else:
			dist[1] == 999999

		return group, dist

	def farest(self,x):
		farx, max_dist = None, 0
		for xi in x:
			dist = self.euclidean_dist(xi,self.weight)
			if max_dist < dist:
				max_dist = dist
				farx = xi
		return farx

	def nearest(self,x):
		nearx, min_dist = None, 99999999
		for xi in x:
			dist = self.euclidean_dist(xi,self.weight)
			if min_dist > dist:
				min_dist = dist
				nearx = xi
		return nearx

	def hebbian_train(self,x,y,eta=1):
		result = []

		# a. checking the current group distances
		yp = self.sigmoid_classify(x)
		group, dist = self.group_distances(x,yp)

		# b. minimizing the distance
		prev_dist = 999999
		while(abs(sum(dist - prev_dist)) > 0.1):
			prev_dist = dist

			if len(group[0])*len(group[1]) > 0:
				if len(group[0]) < len(group[1]):
					coef = float(len(group[1]))/float(len(group[0]))
				else:
					coef = float(len(group[0]))/float(len(group[1]))
			elif len(group[0]) > 0:
				coef = len(group[0])
			else:
				coef = len(group[1])

			print coef
			self.weight = self.weight +\
				eta*(self.farest(x) - self.nearest(x))*(coef)
			self.bias = self.bias + eta*(coef)

			yp = self.sigmoid_classify(x)
			group, dist = self.group_distances(x,yp)
			result.append(abs(sum(y - yp)))
			print dist,self.weight,self.bias

		return result

