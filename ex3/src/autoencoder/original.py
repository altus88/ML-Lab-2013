'''
Created on May 5, 2013

@author: gena
'''

import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import RandomizedPCA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
import cPickle,gzip
import numpy as np
# mnist = fetch_mldata("MNIST original")
# X_train, y_train = mnist.data[:60000] / 255., mnist.target[:60000]
#  
# X_train, y_train = shuffle(X_train, y_train)
# X_train, y_train = X_train[:1000], y_train[:1000] # lets subsample a bit for a first impression
f = open("/home/gena/lab/data/cifar-10-batches-py/data_batch_1","rb") 
dict = cPickle.load(f)
data = dict['data']
y = np.asarray(dict['labels'])
        
X = np.reshape(data,(len(data),3,1024))
X = np.mean(X,1)/float(255)   
    
ind1 = range(0,np.size(X,0))
np.random.shuffle(ind1) 
X_train = X[ind1][:1000]
y_train = y[ind1][:1000] 
classes = ['plane','auto','bird','cat','deer','dog','frog','horse','ship','truck']
   
 
pca = RandomizedPCA(n_components=2)
fig, plots = plt.subplots(10, 10)
fig.set_size_inches(50, 50)
plt.prism()
for i, j in product(xrange(10), repeat=2):
    if i > j:
        continue
    X_ = X_train[(y_train == i) + (y_train == j)]
    y_ = y_train[(y_train == i) + (y_train == j)]
    X_transformed = pca.fit_transform(X_)
    plots[i, j].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
    plots[i, j].set_xticks(())
    plots[i, j].set_yticks(())
    plots[j, i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
    plots[j, i].set_xticks(())
    plots[j, i].set_yticks(())
        
    if i==0:
        
        plots[i, j].set_title(classes[j])
        plots[j, i].set_ylabel(classes[j])
    #plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
#plt.tight_layout()
plt.show()