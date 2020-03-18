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
            y.append(v)
        y = np.array(y)
        return y

    def lse(self,y,yp):
        error = sum((y - yp)**2)
        return error

    def update(self,x,y,yp,eta):
        delta = y - yp
        self.weight = self.weight + eta*x.T.dot(delta)
        self.bias = self.bias + eta*sum(delta)
    
    def train(self,x,y,eta=1,precision=0.00001):
        result = []
        it, error, p = 1, 999999, 999999
        while(p > precision):
            # a. updating previous error
            p_error = error

            # b. classifying
            yp = self.classify(x)
            #print yp

            # c. calculating the least squared error
            error = self.lse(y,yp)
            result.append(error)

            # d. checking the precision
            p = abs(error - p_error)
            #print it,':',p,'( w:',self.weight,', b:',self.bias,')'

            # e. updating the weights
            self.update(x,y,yp,eta)
            it += 1   
        return result

    def sigmoid_classify(self,x,alpha=1):
        y = []
        for sample in x:
            v = np.dot(sample,self.weight) + self.bias
            y.append(1/(1 + exp(-alpha*v)))
        y = np.array(y)
        return y

    def sigmoid_train(self,x,y,eta=1,precision=0.00001):
        result = []
        it, error, p = 1, 999999, 999999
        while(p > precision):
            # a. updating previous error
            p_error = error

            # b. classifying
            yp = self.classify(x)
            #print yp

            # c. calculating the least squared error
            error = self.lse(y,yp)
            result.append(error)

            # d. checking the precision
            p = abs(error - p_error)
            #print it,':',p,'( w:',self.weight,', b:',self.bias,')'

            # e. updating the weights
            self.update(x,y,yp,eta)
            it += 1   
        return result