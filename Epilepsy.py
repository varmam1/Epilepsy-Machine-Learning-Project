from mygrad import Tensor
from mygrad.nnet.layers import dense
from mygrad.nnet.losses import multiclass_hinge
import csv
import numpy as np

def sgd(param, rate):
    """ Performs a gradient-descent update on the parameter.
    
        Parameters
        ----------
        param : mygrad.Tensor
            The parameter to be updated.
        
        rate : float
            The step size used in the update"""
    param.data -= rate*param.grad
    return None

def compute_accuracy(model_out, labels):
    """ Computes the mean accuracy, given predictions and true-labels.
        
        Parameters
        ----------
        model_out : numpy.ndarray, shape=(N, K)
            The predicted class-scores
        labels : numpy.ndarray, shape=(N, K)
            The one-hot encoded labels for the data.
        
        Returns
        -------
        float
            The mean classification accuracy of the N samples."""
    return np.mean(np.argmax(model_out, axis=1) == np.argmax(labels, axis=1))

with open('data.csv', newline='') as csvfile:
	d = list(csv.reader(csvfile, delimiter=' ', quotechar='|'))
	#print(type(d[0][0]))
	data = []
	for i in range(len(d)):
		a = d[i][0].split(",")
		for j in range(len(a)):
			a[j] = int(a[j])
		data.append(a)
	del d

data = np.array(data)
ytrain = data[:,-1]
oneHotEnc = np.zeros((len(ytrain), 5))
for i in range(len(ytrain)):
    oneHotEnc[i][ytrain[i]-1] = 1
ytest = oneHotEnc[10000:]
ytrain = oneHotEnc[:10000]
xtrain = data[:,:-1]
xtest = xtrain[10000:]
xtrain = xtrain[:10000]
del data

D = len(xtrain[0])
K = 5

W = Tensor(np.random.randn(D, K))
b = Tensor(np.zeros((K,), dtype=W.dtype))

l = []
acc = []

params = [b, W]
rate = .1
y = np.argmax(ytrain, axis=1)
for i in range(1000):
    o = dense(xtrain, W) + b
    
    loss = multiclass_hinge(o, y)
    
    l.append(loss.data.item())
    loss.backward()

    acc.append(compute_accuracy(o.data, ytrain))
    
    for param in params:
        sgd(param, rate)
    
    loss.null_gradients()

print(acc[-1])