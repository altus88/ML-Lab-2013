'''
Created on Apr 26, 2013

@author: gena
'''
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import RandomizedPCA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
 
mnist = fetch_mldata("MNIST original")
X_train, y_train = mnist.data[:60000] / 255., mnist.target[:60000]
 
X_train, y_train = shuffle(X_train, y_train)
X_train, y_train = X_train[:100], y_train[:100] # lets subsample a bit for a first impression
 
pca = RandomizedPCA(n_components=784)

X_ = X_train[(y_train == 1)]
y_ = y_train[(y_train == 1)]
X_transformed = pca.fit_transform(X_)


fig, plots = plt.subplots(1, 2)
# 
# 
# X_transformed =  np.dot(V,x_)

immap = 'gray'
plots[0].imshow(np.reshape(X_[0],(28,28)),immap)
plots[1].imshow(np.reshape(X_transformed[0,:],(28,28)),immap)
plt.show()


