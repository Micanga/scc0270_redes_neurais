#####
# IMPORTS
#####
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time

from mnist import MNIST
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from sklearn.model_selection import learning_curve,ShuffleSplit,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tensorflow.keras as keras

#####
# CODE METHODS
#####
def load_mnist(srcdir):
	mndata = MNIST(srcdir)

	X_train, y_train = mndata.load_training()
	X_test, y_test = mndata.load_testing()

	X, y = [], []
	for i in range(len(X_train)):
		X.append(X_train[i])
		y.append(y_train[i])
	for i in range(len(X_test)):
		X.append(X_test[i])
		y.append(y_test[i])

	return X, y

def plot_mnist_stratification(y,plot_name,show=False):
	plt.figure(figsize=(12,6));

	x = np.arange(10)
	count = [list(y).count(i) for i in range(10)]
	count = count/np.array(count).sum()
	plt.barh(x, width=count)
	plt.xlabel('Percentual de amostras',fontsize='x-large')
	plt.ylabel('Classe',fontsize='x-large')
	plt.yticks(x)

	plt.savefig('plots/'+plot_name+'.pdf',bbox_inches='tight')
	if show:
		plt.show()

def get_class_examples(X,y):
	figures = []
	choosed_label = []
	for i in range(len(X)):
		if y[i] not in choosed_label:
			figures.append(x[i])
			choosed_label.append(y[i])
		if len(figure) == 10:
			break
	return figures

def plot_mnist_figure(figures,plot_name,plot_shape,show=False):
	plt.figure(figsize=(12,6));
	for i in range(len(figures)):
		plt.subplot(plot_shape[0], plot_shape[1], i+1);
		plt.imshow(np.array(figures[i]).reshape(28,28),
		              cmap = plt.cm.gray, interpolation='nearest',
		              clim=(0, 255));

	plt.savefig('plots/'+plot_name+'.pdf',bbox_inches='tight')
	if show:
		plt.show()

def plot_pca_variance_ratio(pca,plot_name,show=False):
	plt.figure(figsize=(12,8));
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('Número de componentes',fontsize='x-large')
	plt.ylabel('Quantidade de Informação Cumulativa',fontsize='x-large')

	plt.savefig('plots/'+plot_name+'.pdf',bbox_inches='tight')
	if show:
		plt.show()

def init_cnn(n_layers, num_classes, filter_sizes, alpha,\
input_shape=(28,28,1), kernel_size=(3,3), pool_size=(2,2), dropout=False, dropout_rate=0.3):
	model = keras.Sequential()

	# Input Layer
	model.add(keras.layers.Conv2D(filters=filter_sizes[0], kernel_size=kernel_size,activation='linear',\
								input_shape=input_shape,padding='same'))
	model.add(keras.layers.LeakyReLU(alpha=alpha))
	model.add(keras.layers.MaxPooling2D(pool_size,padding='same'))
	if dropout:       
		if isinstance(dropout_rate, list):
			model.add(keras.layers.Dropout(dropout_rate[0]))
		else:
			model.add(keras.layers.Dropout(dropout_rate))

	# Mid Layers
	for i in range(1,n_layers-1):
		model.add(keras.layers.Conv2D(filter_sizes[i], kernel_size, activation='linear',padding='same'))
		model.add(keras.layers.LeakyReLU(alpha=alpha))
		model.add(keras.layers.MaxPooling2D(pool_size,padding='same'))
		if dropout:           
			if isinstance(dropout_rate, list):
				model.add(keras.layers.Dropout(dropout_rate[i]))
			else:
				model.add(keras.layers.Dropout(dropout_rate))

	# Output Layer
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(filter_sizes[-1], activation='linear'))
	model.add(keras.layers.LeakyReLU(alpha=alpha))          
	if dropout:           
		if isinstance(dropout_rate, list):
			model.add(keras.layers.Dropout(dropout_rate[-1]))
		else:
			model.add(keras.layers.Dropout(dropout_rate))
	model.add(keras.layers.Dense(num_classes, activation='softmax'))

	# Compiling
	model.compile(loss=keras.losses.categorical_crossentropy,\
	 optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

	# Returning
	return model

def plot_cnn_result(cnn,plot_name,show=False):
	accuracy = cnn.history['accuracy']
	val_accuracy = cnn.history['val_accuracy']
	epochs = range(len(accuracy))

	plt.figure(figsize=(12,8));
	plt.plot(epochs, accuracy, 'bo', label='Acurácia de Treino')
	plt.plot(epochs, val_accuracy, 'b', label='Acurácia de Validação')
	plt.legend()

	plt.savefig('plots/'+plot_name+'.pdf',bbox_inches='tight')
	if show:
		plt.show()

def plot_cnn_result_clean(accuracy,val_accuracy,plot_name,show=False):
	epochs = range(len(accuracy))

	plt.figure(figsize=(12,8));
	plt.plot(epochs, accuracy, 'bo', label='Acurácia de Treino')
	plt.plot(epochs, val_accuracy, 'b', label='Acurácia de Validação')
	plt.legend()

	plt.savefig('plots/'+plot_name+'.pdf',bbox_inches='tight')
	if show:
		plt.show()

def plot_learning_curve(classifier, X, y, cv=5,train_sizes=np.linspace(.1, 1.0, 5),show=False):
    train_sizes, train_scores, test_scores = \
        learning_curve(classifier, X, y, cv=cv, train_sizes=train_sizes,verbose=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.figure(figsize=(12,8))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")

    plt.savefig('plots/svm_curve.pdf',bbox_inches='tight')
    if show:
        plt.show()

    return plt

#####
# MAIN CODE
#####
# 1. Loading MNIST dataset
# a. loading the original data
print('Loading the dataset...')
X,y = load_mnist('mnist_dataset')

plot_mnist_stratification(y,'mnist_class_distribution')
figures = get_class_examples(X_train,y_train)
plot_mnist_figure(figures,'mnist_dataset_example',(2,5),False)

# 2. Pre-processing
print('Pre-processing...')
if os.path.exists("mnist_dataset/cnn_X.Pickle")\
and os.path.exists("mnist_dataset/svm_X.Pickle"):
	# loading the pre processed data
	with open("mnist_dataset/cnn_X.Pickle","rb") as file:
		X_cnn = pickle.load(file)
		y_cnn = np.array(keras.utils.to_categorical(y))
	with open("mnist_dataset/svm_X.Pickle","rb") as file:
		X_svm = pickle.load(file)
		y_svm = np.array(y)
		print('SVM DataShape:',X_svm.shape,y_svm.shape)
else:
	# a. scaling the data via mean and std dev scaler
	print('1) Scaling...')
	scaler = StandardScaler()	
	X_scaled = scaler.fit_transform(X,y)
	print('Dataset shape:',X_scaled.shape)

	# b. reshaping for cnn
	X_cnn = X_scaled.reshape(-1, 28,28, 1)
	y_cnn = np.array(keras.utils.to_categorical(y))

	# c. saving the cnn processed dataset
	with open("mnist_dataset/cnn_X.Pickle","wb") as file:
		pickle.dump(X_cnn,file)

	# d. applying PCA method
	print('2) Applying PCA...')
	pca = PCA(.9)
	X_svm = pca.fit_transform(X_scaled,y)
	y_svm = np.array(y)
	print('Dataset shape:',X_svm.shape)

	# e. ploting the pca variance ratio
	plot_pca_variance_ratio(pca,'pca_variance_ratio')

	# f. saving the svm processed dataset
	with open("mnist_dataset/svm_X.Pickle","wb") as file:
		pickle.dump(X_svm,file)

# 3. Initialising the classifiers, training and collecting the results
# a. cnn
cnn = init_cnn(4, 10, [32,64,128,128], 0.1,\
	input_shape=(28,28,1), kernel_size=(3,3), pool_size=(2,2),\
	 dropout=True,dropout_rate=[0.25,0.25,0.4,0.3])

# - training
print('CNN Fitting')
X_cnn_train, X_cnn_test, y_cnn_train, y_cnn_test =\
	train_test_split(X_cnn, y_cnn, test_size=0.15, random_state=1)
if not os.path.exists("cnn.h5py"):
	cnn_history = cnn.fit(X_cnn_train, y_cnn_train, \
		batch_size=64,epochs=20,validation_data=(X_cnn_test, y_cnn_test))
	cnn.save("cnn.h5py")

	plot_cnn_result(cnn_history,'cnn_curve')
else:
	cnn = keras.models.load_model('cnn.h5py')
	accuracy, val_accuracy = [], []
	with open('clean_output.txt','r') as file:
		for line in file:
			data = line.split(',')
			accuracy.append(float(data[0]))
			val_accuracy.append(float(data[1]))
	plot_cnn_result_clean(np.array(accuracy),np.array(val_accuracy),'cnn_curve')

# - showing the result
test_eval = cnn.evaluate(X_cnn_test, y_cnn_test)
print('CNN - Test loss:', test_eval[0])
print('CNN - Test accuracy:', test_eval[1])

predicted_classes = cnn.predict(X_cnn_test)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
y_cnn_test = np.argmax(np.round(y_cnn_test),axis=1)

target_names = ["Class {}".format(i) for i in range(10)]
print(classification_report(y_cnn_test, predicted_classes, target_names=target_names))

# b. svm
svm = SVC(C=0.01, kernel='linear', gamma='scale',\
	coef0=0.0, shrinking=True, probability=True, tol=0.001,\
	 cache_size=80000, class_weight=None, verbose=False, max_iter=-1,\
	  decision_function_shape='ovr', random_state=None)

# - training
print('SVM Fitting')
X_svm_train, X_svm_test, y_svm_train, y_svm_test =\
	train_test_split(X_svm, y_svm, test_size=0.15, random_state=1)
print(X_svm_train.shape, X_svm_test.shape, y_svm_train.shape, y_svm_test.shape)

# - showing the result
print('- Predict')
start_time = time.time()
svm.fit(X_svm_train,y_svm_train)
print("--- %s seconds ---" % (time.time() - start_time))
predicted_classes = svm.predict(X_svm_test)

target_names = ["Class {}".format(i) for i in range(10)]
print(accuracy_score(y_svm_test,predicted_classes))
print(hamming_loss(y_svm_test,predicted_classes))
print(classification_report(y_svm_test, predicted_classes, target_names=target_names))

print('- Learning curve')
plot_learning_curve(svm,X_svm_train,y_svm_train)
