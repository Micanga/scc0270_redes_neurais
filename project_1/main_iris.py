#####
# IMPORTS
#####
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier

#####
# SUPPORT VARIABLES
#####
COLORS = ['red','green','blue']
LABELS = ['Setosa', 'Versic.', 'Virgin.']

#####
# CODE METHODS
#####
def load_iris():
	iris = datasets.load_iris()
	X, y = iris.data, iris.target
	return X, y

def plot_iris_data(X,y,plot_name='figure',show=False):
	# 1. Initialising the figure and the support variables
	fig, axs = plt.subplots(4,4,figsize=(12,12))
	c = [COLORS[int(y[k])] for k in range(len(y))]

	# 2. Building the plot
	for i in range(len(axs)):
		for j in range(len(axs)):
			axs[i,j].scatter(X[:,j],X[:,i],s=10.0, color=c)
			if i == 3:
				axs[i,j].set_xlabel('Parametro '+str(j+1),fontsize='x-large')
			if j == 0:
				axs[i,j].set_ylabel('Parametro '+str(i+1),fontsize='x-large')

	# 3. Saving the figure
	plt.savefig('plots/'+plot_name+'.pdf',bbox_inches='tight')
	if show:
		plt.show()

def mean_std_normalisation(x):
	return (np.array(x) - np.array(x).mean())/np.array(x).std()

def kfold_cross_validation(X,y,classifiers,cv=5,repeat=10):
	train_scores = [[] for i in range(len(classifiers))]
	test_scores  = [[] for i in range(len(classifiers))]
	estimators = [[] for i in range(len(classifiers))]

	for r in range(repeat):
		print('Cross validation',r+1,'/',repeat)

		for i in range(len(classifiers)):
			cv_result = cross_validate(classifiers[i], X, y, cv=cv,\
							return_train_score=True, return_estimator=True)
			train_scores[i].append(np.array(cv_result['train_score']).mean())
			test_scores[i].append(np.array(cv_result['test_score']).mean())


			max_index = list(cv_result['train_score']).index(max(cv_result['train_score']))
			estimators[i].append(cv_result['estimator'][max_index])

	return train_scores, test_scores, estimators

def plot_confusion_matrix(X,y,classifier,plot_name='figure',show=False):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	cm = confusion_matrix(y, classifier.predict(X))
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	cax = ax.matshow(cm)

	for (i, j), z in np.ndenumerate(cm):
	    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
	            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

	ax.set_xticklabels([''] + LABELS)
	ax.set_yticklabels([''] + LABELS)
	if i == 0:
		ax.set_xlabel('Predicted')
	ax.set_ylabel('True')

	plt.savefig('plots/'+plot_name+'.pdf',bbox_inches='tight')
	if show:
		plt.show()

#####
# MAIN CODE
#####
# 1. Loading the iris dataset
X,y = load_iris()	
#plot_iris_data(X,y,'iris_distribution')

# 2. Preprocessing the data
X = mean_std_normalisation(X)
#plot_iris_data(X,y,'iris_distribution_normal')

# 3. Defining two classifiers
mlps = []
mlps.append(MLPClassifier(hidden_layer_sizes=(3,3, ), activation='logistic',\
	solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant',\
	learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None,\
	tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,\
	early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,\
	n_iter_no_change=10))

mlps.append(MLPClassifier(hidden_layer_sizes=(4,3,3, ), activation='logistic',\
	solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant',\
	learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None,\
	tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,\
	early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,\
	n_iter_no_change=10))

# 4. Applying the Stratified K-Fold validation (K=5)
# for both defined models and getting result data
train_scores, test_scores, estimators = kfold_cross_validation(X,y,mlps)

# 5. Printing the result
for i in range(len(train_scores)):
	print('===== MLP',i+1)
	print('Test accuracy:',np.array(test_scores[i]).mean(), np.array(test_scores[i]).std())
	print('Train accuracy:',np.array(train_scores[i]).mean(), np.array(train_scores[i]).std())

# 6. Plotting the confusion matrixs
for i in range(len(estimators)):
	for j in range(len(estimators[i])):
		plot_confusion_matrix(X,y,estimators[i][j],'iris_cm_mlp_'+str(i+1)+'_'+str(j))