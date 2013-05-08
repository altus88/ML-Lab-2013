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
    
#img1 = dict['data']
 
#ind = (np.array(dict['labels']) == 1) 
 
#X = np.reshape(img1[ind],(np.size(np.where(ind==True)),3,1024))
X = np.reshape(img,(len(img),3,1024))

nfeatures = 14*14

K = 500
num_iter = 100

X = np.mean(X,1)
X,mu,std = featureNormilize(X)

def rescale(X):
    return rescaleImage(X, (32,32), (14,14))

X_ = np.array(map(rescale,X))

print "whitening"
X_ = whitening(X_,0.01)

initial_centroids = randomSampleFromData(X_,K)
batch_size = 1000
num_iters = 5

#c = trainMiniBatchK_means(np.mat(X_),initial_centroids,batch_size,num_iters)

#mbk = MiniBatchKMeans(init='k-means++', n_clusters=K, batch_size=batch_size,
#                      n_init=num_iters, max_no_improvement=10, verbose=1)

#mbk.fit(X_)
#print "K = " +  str(np.size(c,1))
k_means = KMeans(init='random', n_clusters=K,max_iter = 15, n_init=1,verbose=1)
#t0 = time.time()
k_means.fit(X_)

mbk_means_labels = k_means.labels_
print str(len(np.unique(mbk_means_labels)))
c = k_means.cluster_centers_.T
visualizeFilters(c)  
raw_input("Press ENTER to exit")  


