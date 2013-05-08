'''
Created on Apr 25, 2013

@author: gena
'''

from utils import *

img = np.array
nBatches = 5
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

K = 400
num_iter = 15

X = np.mean(X,1)
X,mu,std = featureNormilize(X)

def rescale(X):
    return rescaleImage(X, (32,32), (14,14))

X_ = np.array(map(rescale,X))

print "whitening"
X_ = whitening(X_,0.01)


# cov = np.dot(X.T,X)/len(X)
#plt.imshow(cov)
# plt.figure(1)
# plt.imshow(np.reshape(X[0],(32,32)),'gray')
# plt.figure(2)
# 
# plt.imshow(np.reshape(X_[0],(14,14)),'gray')
# plt.show()



# plt.figure()
# plt.imshow(np.reshape(X[0].T,(32,32)),'gray')
# plt.show()

initial_centroids = initializeClusters(nfeatures,K)
print "run k-means"
c,v = runK_means(np.mat(X_),initial_centroids,K,num_iter)   

visualizeFilters(c)   
raw_input("Press ENTER to exit")      
        