import tensorflow.keras as keras

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