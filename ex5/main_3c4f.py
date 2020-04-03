import matplotlib.pyplot as plt
import numpy as np
from re import match
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, confusion_matrix
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

# 2. Initialising 2 differents MLPs
# NOTE: We use the Quasi-Newton optimiser because
# of the dataset size (best choice for convergence) 
mlp = []

# a. MLPClassifier with:
# > architecture: 4:8:8:8:1
# - 3 hidden_layer and 8:8:8 neurons;
# - logistic activation function, and;
# - quasi-Newton optimiser.
mlp.append(MLPClassifier(hidden_layer_sizes=(8,8,8,),\
						 activation='logistic',\
						 solver='lbfgs',
						 random_state=1))

# b. MLPClassifier with:
# > architecture: 4:4:1
# - 1 hidden_layers and 4 neurons;
# - identity activation function, and;
# - quasi-Newton optimiser.
mlp.append(MLPClassifier(hidden_layer_sizes=(4,),\
						 activation='identity',\
						 solver='lbfgs',\
						 random_state=1))

# 3. Fitting the dataset
for i in range(2):
	print "Fitting MLP",i+1
	mlp[i].fit(X_train, Y_train)

# 4. Classifying
cm = []
for i in range(2):
	print "=====\n| MLP",i+1,"\n====="
	print "Training set score: %f" % mlp[i].score(X_train, Y_train)
	print "Test set score: %f" % mlp[i].score(X_test, Y_test)
	print "Generalised score: %f" % (mlp[i].score(X_test, Y_test) - mlp[i].score(X_train, Y_train))
	print "Precision score:", precision_score(Y_test, mlp[i].predict(X_test), average= None)
	cm.append(confusion_matrix(Y_test, mlp[i].predict(X_test)))
	cm[i] = cm[i].astype('float') / cm[i].sum(axis=1)[:, np.newaxis]

# 5. Printing the confusion matrix
labels = ['Class 1','Class 2','Class 3']
fig = plt.figure()

for i in range(2):
	ax = fig.add_subplot(211+i)
	cax = ax.matshow(cm[i])

	for (i, j), z in np.ndenumerate(cm[i]):
	    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
	            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

	ax.set_xticklabels([''] + labels)
	ax.set_yticklabels([''] + labels)
	if i == 0:
		ax.set_xlabel('Predicted')
	ax.set_ylabel('True')
plt.show()

print '=====\n| BEST ARCHITECTURE:\n=====\n',mlp[1]