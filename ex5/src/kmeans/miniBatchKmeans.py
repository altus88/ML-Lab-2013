'''
Created on Apr 29, 2013

@author: gena
'''
from utils import *
from sklearn.cluster import MiniBatchKMeans, KMeans
import time

img = np.array
nBatches = 1
for i in range(1,nBatches+1):
    f = open("/home/gena/lab/data/cifar-10-batches-py/data_batch_"+ str(i),"rb") 
    dict = cPickle.load(f)
    f.close() 
    if i!=1:
        img = np.append(img,dict['data'],0)
    else:
        img = dict['data']    
    
#img1 = dict['data']
 
#ind = (np.array(dict['labels']) == 1) 
 
#X = np.reshape(img1[ind],(np.size(np.where(ind==True)),3,1024))
X = np.reshape(img,(len(img),3,1024))

nfeatures = 14*14

K = 256
num_iter = 10

X = np.mean(X,1)
X,mu,std = featureNormilize(X)

def rescale(X):
    return rescaleImage(X, (32,32), (14,14))

X_ = np.array(map(rescale,X))

print "whitening"
X_ = whitening(X_,0.01)

initial_centroids = initializeClusters(nfeatures,K)
batch_size = 200
num_iters = 100

c = trainMiniBatchK_means(X_,initial_centroids,batch_size,num_iters)




