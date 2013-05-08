'''
Created on Apr 26, 2013

@author: gena
'''
import cPickle,gzip
import numpy as np
import utils as ut
from autoencoder import *
from scipy.optimize import fmin_bfgs  ,fmin_l_bfgs_b
import matplotlib.pyplot as plt


if 1:
    #Visualize MNIST data
    path = '/home/gena/lab/data/mnist.pkl.gz'
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    ind1 = range(0,np.size(train_set[0],0))
    np.random.shuffle(ind1) 
    X = train_set[0][ind1][:1000]
    y = train_set[1][ind1][:1000]
    scatterPlot2D(X,y,10,range(10))

if 0:
    #Visualize CIFAR data
    path = "/home/gena/lab/data/cifar-10-batches-py/data_batch_1"
    f = open(path,"rb") 
    dict = cPickle.load(f)
    data = dict['data']
    y = np.asarray(dict['labels'])
        
    X = np.reshape(data,(len(data),3,1024))
    X = np.mean(X,1)/float(255)   
    
    ind1 = range(0,np.size(X,0))
    np.random.shuffle(ind1) 
    X = X[ind1][:1000]
    y = y[ind1][:1000]
    classes = ['plane','auto','bird','cat','deer','dog','frog','horse','ship','truck']
    scatterPlot2D(X,y,10,classes)

