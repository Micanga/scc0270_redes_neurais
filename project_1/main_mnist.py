#####
# IMPORTS
#####
import matplotlib.pyplot as plt
from mnist import MNIST
import numpy as np
from sklearn.decomposition import PCA
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

	return X_train, y_train, X_test, y_test

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

#####
# MAIN CODE
#####
# 1. Loading MNIST dataset
# a. loading the original data
"""
X_train, y_train, X_test, y_test = load_mnist('mnist_dataset')

figures = []
choosed_label = []
for i in range(len(X_train)):
	if y_train[i] not in choosed_label:
		figures.append(X_train[i])
		choosed_label.append(y_train[i])
plot_mnist_figure(figures,'mnist_dataset_example',(2,5),False)

# 2. Pre-processing
# a. scaling the data via mean and std dev scaler
scaler = StandardScaler()	
X_train = scaler.fit_transform(X_train,y_train)
X_test  = scaler.fit_transform(X_test, y_test)
print('Train shape:',X_train.shape,'/Test shape:',X_test.shape)

# b. applying PCA method
pca = PCA(.95)
X_train = pca.fit_transform(X_train,y_train)
X_test  = pca.fit_transform(X_test, y_test)
print('Train shape:',X_train.shape,'/Test shape:',X_test.shape)
"""
# 3. Initialising the classifiers
# a. cnn
cnn = init_cnn(4, 10, [32,64,128,128], 0.1,\
	input_shape=(28,28,1), kernel_size=(3,3), pool_size=(2,2),\
	 dropout=True,dropout_rate=[0.25,0.25,0.4,0.3])

# b. svm
svm = SVC(C=1.0, kernel='poly', degree=9, gamma='scale',\
	coef0=0.0, shrinking=True, probability=True, tol=0.001,\
	 cache_size=200, class_weight=None, verbose=False, max_iter=-1,\
	  decision_function_shape='ovr', random_state=None)

# 4. Training
"""
cnn_trained = cnn.fit(X_train, y_train, \
	batch_size=64,epochs=20,validation_data=(X_test, y_test))
svm.fit(X_train,y_train)

# 5. Plotting result
accuracy = cnn_trained.history['acc']
val_accuracy = cnn_trained.history['val_acc']
loss = cnn_trained.history['loss']
val_loss = cnn_trained.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()	
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
"""