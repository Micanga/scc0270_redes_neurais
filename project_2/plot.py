import matplotlib.pyplot as plt
import numpy as np

def plot_class_stratification(y,plot_name,show=False):
	plt.figure(figsize=(12,6));

	# counting the classes
	n_classes = len(np.unique(y))
	count = [list(y).count(i) for i in range(10)]

	# plotting
	x = np.arange(n_classes)
	plt.barh(x, width=count)
	plt.xlabel('Número de amostras',fontsize='x-large')
	plt.ylabel('Classe',fontsize='x-large')
	plt.yticks(x)

	# saving the plot
	plt.savefig('plots/'+plot_name+'.pdf',bbox_inches='tight')
	if show:
		plt.show()

def get_class_examples(X,y):
	examples, choosed_label = [], []
	n_classes = len(np.unique(y))

	for i in range(len(X)):
		if y[i] not in choosed_label:
			examples.append(X[i])
			choosed_label.append(y[i])

		if len(examples) == n_classes:
			break

	return examples

def plot_class_examples(X,y,plot_name,show=False):
	# getting class examples
	examples = get_class_examples(X,y)

	# plotting
	plt.figure(figsize=(12,6));
	for i in range(len(examples)):
		plt.subplot(2, len(examples)/2, i+1);
		plt.imshow(np.array(examples[i]).reshape(28,28),
		              cmap = plt.cm.gray, interpolation='nearest',
		              clim=(0, 255));

	# saving the plot
	plt.savefig('plots/'+plot_name+'.pdf',bbox_inches='tight')
	if show:
		plt.show()

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