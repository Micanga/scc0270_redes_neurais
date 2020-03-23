from math import *
from numpy.random import uniform
import numpy as np


class Adaline:

    def __init__(self,nweights,bias=None):
        self.weight = uniform(-1,1,nweights)
        self.bias = (uniform(-1,1,1))[0] if bias is None else bias
    
    def classify(self,x):
        y = []
        for sample in x:
            v = np.dot(sample,self.weight) + self.bias
            if v <= 0:
                y.append(0)
            elif v >= 1:
                y.append(1)
            else:
                y.append(v)
        y = np.array(y) + 1
        return y

    def lse(self,y,yp):
        error = sum((y - yp)**2)
        return error

    def update(self,x,y,yp,eta):
        delta = y - yp
        self.weight = self.weight + eta*x.T.dot(delta)
        self.bias = self.bias + eta*sum(delta)
    
    def train(self,x,y,eta=1,precision=0.01,max_it=500):
        result = []
        it, error = 1, 999999
        while(error > precision and it < max_it):
            # a. classifying
            yp = self.classify(x)
            #print yp

            # b. calculating the least squared error
            error = self.lse(y,yp)
            result.append(error)

            # c. updating the weights
            self.update(x,y,yp,eta)
            it += 1   
        return result

    def sigmoid_classify(self,x,alpha=1):
        y = []
        for sample in x:
            v = np.dot(sample,self.weight) + self.bias
            if v <= 0:
                y.append(0)
            elif v >= 1:
                y.append(1)
            else:
                y.append(1/(1 + exp(-alpha*v)))
        y = np.array(y) + 1
        return y

    def sigmoid_train(self,x,y,eta=1,precision=0.001,max_it=500):
        result = []
        it, error = 1, 999999
        while(error > precision and it < max_it):
            # a. classifying
            yp = self.classify(x)
            #print yp

            # b. calculating the least squared error
            error = self.lse(y,yp)
            result.append(error)

            # c. updating the weights
            self.update(x,y,yp,eta)
            it += 1   
        return result