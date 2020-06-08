#####
# IMPORTS
#####
from datetime import datetime
import os

from sklearn.metrics import classification_report
import tensorflow.keras as keras

from cnn import *
from plot import *

#####
# MAIN CODE
#####
# 1. Loading Fashion-MNIST dataset
# a. loading the original data
print('Loading the dataset...')
(X_train,y_train), (X_test,y_test) = keras.datasets.fashion_mnist.load_data()

# b. ploting some information about the dataset
plot_class_stratification(y_train,'class_distribution')
plot_class_examples(X_train,y_train,'class_examples')

# 2. Pre-processing
print('Pre-processing...')

# a. reshaping the parameters
X_train = X_train.reshape(-1, 28,28, 1)
X_test = X_test.reshape(-1, 28,28, 1)

# b. transforming to floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# c. scaling
X_train = X_train/255.0
X_test = X_test/255.0

# d. transforming classes into hotkeys
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# 3. Initialising the classifiers, training and collecting the results
# a. cnn
cnn = init_cnn(4, 10, [32,64,128,128], 0.1,\
	input_shape=(28,28,1), kernel_size=(3,3), pool_size=(2,2),\
	 dropout=True,dropout_rate=[0.25,0.25,0.4,0.3])

# - training
print('Fitting...')
if not os.path.exists("cnn.h5py"):
	start_time = datetime.now()
	cnn_history = cnn.fit(X_train, y_train, \
		batch_size=64,epochs=10,validation_data=(X_test, y_test))
	print('Time Delta:',datetime.now()  -start_time)
	cnn.save("cnn.h5py")
	plot_cnn_result(cnn_history,'cnn_curve')
else:
	cnn = keras.models.load_model('cnn.h5py')

# - showing the result
test_eval = cnn.evaluate(X_test, y_test)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

predicted_classes = cnn.predict(X_test)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
y_test = np.argmax(np.round(y_test),axis=1)

target_names = ["Class {}".format(i) for i in range(10)]
print(classification_report(y_test, predicted_classes, target_names=target_names))
