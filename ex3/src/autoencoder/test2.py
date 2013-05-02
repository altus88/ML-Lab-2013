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


print "download data"
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()



ind1 = range(0,np.size(train_set[0],0))
ind2 = range(0,np.size(valid_set[0],0))
ind3 = range(0,np.size(test_set[0],0))

np.random.shuffle(ind1) 
np.random.shuffle(ind2) 
np.random.shuffle(ind3) 

X = train_set[0][ind1]
y = train_set[1][ind1]
 
X_v = valid_set[0][ind2]
y_v = valid_set[1][ind2]
 
X_t = test_set[0][ind3]
y_t = test_set[1][ind3]

print "PCA"


x_ = X[0]
#x_-=np.mean(x_)


#m = size(X)
   
U,S,V =  np.linalg.svd(np.dot(np.mat(x_).T,np.mat(x_)))
X_transformed = np.dot(x_,V[:,:784])
# cov = np.mat(x_).T * x_
# 
# U,S,V =  np.linalg.svd(cov)
# 
fig, plots = plt.subplots(1, 2)
# 
# 
# X_transformed =  np.dot(V,x_)

immap = 'gray'
plots[0].imshow(np.reshape(x_,(28,28)),immap)
plots[1].imshow(np.reshape(U[:,0],(28,28)),immap)
plt.show()
print "h"
