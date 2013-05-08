'''
Created on Apr 29, 2013

@author: gena
'''
from utils import *
import time
from sklearn.cluster import MiniBatchKMeans, KMeans

img = np.array
nBatches = 5
for i in range(1,nBatches+1):
    with open("/home/gena/lab/data/cifar-10-batches-py/data_batch_"+ str(i),"rb") as f: 
        dict = cPickle.load(f)
    if i!=1:
        img = np.append(img,dict['data'],0)
    else:
        img = dict['data']    
    
X = np.reshape(img,(len(img),3,1024))

nfeatures = 14*14

X = np.mean(X,1)
X,mu,std = featureNormilize(X)

def rescale(X):
    return rescaleImage(X, (32,32), (14,14))

X_ = np.array(map(rescale,X))

print "whitening"
X_ = whitening(X_,0.01)

K = 400
initial_centroids = randomSampleFromData(X_,K)
batch_size = 1000
num_iters = 500

c = trainMiniBatchK_means(np.mat(X_),initial_centroids,batch_size,num_iters)

print str(np.size(c,1))
visualizeFilters(c)  
raw_input("Press ENTER to exit")  


