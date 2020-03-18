import numpy as np
import matplotlib.pyplot as plt
from re import match

from adaline import Adaline
from perceptron import Perceptron

COLORS = ['r','g','b','purple','yellow','black']

# 0. Reading the datasets
# a. Dataset 1
x1, y1 = [], []
with open('Aula3-dataset_1.csv','r') as datafile:
	for line in datafile:
		if match('^\d+.\d+,\d+.\d+,\d+$',line) is not None:
			data = line.split(',')
			x1.append(np.array([float(d) for d in data[:-1]]))
			y1.append(np.array(float(data[-1])))
v11 = np.array([x[0] for x in x1])
v21 = np.array([x[1] for x in x1])

# b. Dataset 2
x2, y2 = [], []
with open('Aula3-dataset_2.csv','r') as datafile:
	for line in datafile:
		if match('^\d+.\d+,\d+.\d+,\d+.\d+,\d+.\d+,\d+$',line) is not None:
			data = line.split(',')
			x2.append(np.array([float(d) for d in data[:-1]]))
			y2.append(np.array(float(data[-1])))
v12 = np.array([x[0] for x in x1])
v22 = np.array([x[1] for x in x1])

# c. converting to np.array
x1, y1 = np.array(x1), np.array(y1)
x2, y2 = np.array(x2), np.array(y2)

# d. Plotting the datasets
"""fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.title.set_text('Dataset 1')

ax2.title.set_text('Dataset 2')

ax1.scatter(v11,v21)
ax2.scatter(v12,v22)

plt.show()"""

# 1. Running the perceptron and adaline training
# a. PERCEPTRON
print '=== PERCEPTRON ====='
# - DATASET 1
errors_d1, etas = [], [0.1,0.3,0.5,0.7,0.9]
for eta in etas:
	perc_d1 = Perceptron(2) 
	print '(D1) Training (eta =',eta,')'
	errors_d1.append(perc_d1.sigmoid_train(x1,y1,eta))
print '(D1) FINISHED\n==='

# - DATASET 2
errors_d2, etas = [], [0.1,0.3,0.5,0.7,0.9]
for eta in etas:
	perc_d2 = Perceptron(4) 
	print '(D2) Training (eta =',eta,')'
	errors_d2.append(perc_d2.sigmoid_train(x2,y2,eta))
print '(D2) FINISHED\n==='

# - plotting the learning rate for perceptron
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plt.suptitle('Perceptron')

for i in range(len(errors_d1)):
	ax1.plot(errors_d1[i],color=COLORS[i],label='eta = '+str(etas[i]))
ax1.title.set_text('Learning Rate (Dataset 1)')

for i in range(len(errors_d2)):
	ax2.plot(errors_d2[i],color=COLORS[i],label='eta = '+str(etas[i]))
ax2.title.set_text('Learning Rate (Dataset 2)')

plt.legend()
plt.show()

# b. ADALINE
print '=== ADALINE ====='
# - DATASET 1
errors_d1, etas = [], [0.1,0.3,0.5,0.7,0.9]
for eta in etas:
	adal_d1 = Adaline(2) 
	print '(D1) Training (eta =',eta,')'
	errors_d1.append(adal_d1.sigmoid_train(x1,y1,eta))
print '(D1) FINISHED\n==='

# - DATASET 2
errors_d2, etas = [], [0.1,0.3,0.5,0.7,0.9]
for eta in etas:
	adal_d2 = Adaline(4) 
	print '(D2) Training (eta =',eta,')'
	errors_d2.append(adal_d2.sigmoid_train(x2,y2,eta))
print '(D2) FINISHED\n==='

# - plotting the learning rate for perceptron
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plt.suptitle('Adaline')

for i in range(len(errors_d1)):
	ax1.plot(errors_d1[i],color=COLORS[i],label='eta = '+str(etas[i]))
ax1.title.set_text('Learning Rate (Dataset 1)')

for i in range(len(errors_d2)):
	ax2.plot(errors_d2[i],color=COLORS[i],label='eta = '+str(etas[i]))
ax2.title.set_text('Learning Rate (Dataset 2)')

plt.legend()
plt.show()