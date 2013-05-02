'''
Created on Apr 28, 2013

@author: gena
'''
from utils import *

f = open("/home/gena/lab/data/cifar-10-batches-py/data_batch_2","rb") 
dict = cPickle.load(f)
f.close() 
 
img1 = dict['data'][1]
 
X = np.reshape(img1,(3,1024)).T

 
X = np.reshape(np.mean(X,1),(32,32))

#data = ds.fetch_olivetti_faces()

#X = data['images'][0]

#X,mu,std = featureNormilize(X,additive = 0)

X_ = whitening(np.mat(X),0.0001)
 
cov = np.dot(X_.T,X_)    


plt.figure(1)
plt.imshow(np.reshape(X,(32,32)),'gray')
plt.figure(2)
plt.imshow(np.reshape(X_,(32,32)),'gray')
plt.figure(3)
plt.imshow(cov)
plt.show()