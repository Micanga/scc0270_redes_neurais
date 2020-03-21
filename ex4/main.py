import numpy as np
from re import match
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# 1. Defining the dataset
# a. reading the data
X, Y = [], []
with open('Dataset_3Cluster_4features.csv','r') as datafile:
	for line in datafile:
		if match('^\d+.\d+,\d+.\d+,\d+.\d+,\d+.\d+,\d+$',line) is not None:
			data = line.split(',')
			X.append(np.array([float(d) for d in data[:-1]]))
			Y.append(float(data[-1]))

# b. spliting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33,random_state=1)

# 2. Initialising 3 differents  MLPs
mlp = []
# a. default SkLearn MLPClassifier
mlp.append(MLPClassifier(random_state=1))

# b. tripled hidden-layer default SkLearn MLPClassifier
mlp.append(MLPClassifier(hidden_layer_sizes=(300,),random_state=1))

# c. stocastic gradient descendent MLPClassifier (alpha=1e-5 and 500 neurons in HL)
mlp.append(MLPClassifier(hidden_layer_sizes=(500,), solver='sgd', alpha=1e-2,random_state=1))

# 3. Fitting the dataset
for i in range(3):
	mlp[i].fit(X_train, Y_train)

# 4. Classifying:
for i in range(3):
	print "=====\n| MLP",i,"\n====="
	print "Training set score: %f" % mlp[i].score(X_train, Y_train)
	print "Test set score: %f" % mlp[i].score(X_test, Y_test)

# 5. Plotting the learning rate
import matplotlib.pyplot as plt
color = ['r','g','b']
title = ['Default','Tripled Default','Stochastic']
fig, axes = plt.subplots(1, 3, figsize=(10, 15))
# load / generate some toy datasets
for i in range(3):
    axes[i].plot(np.array(mlp[i].loss_curve_),color=color[i])
    axes[i].plot(np.array([mlp[i].loss_curve_[-1]\
    	 for j in range(len(mlp[i].loss_curve_))]),linestyle='--',color=color[i])
    axes[i].set_title(title[i])
    axes[i].set_xlabel('Iterations')
    axes[i].set_ylim([0,1.25])

axes[0].set_ylabel('Loss Curve')
plt.show()