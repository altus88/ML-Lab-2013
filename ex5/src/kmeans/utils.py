'''
Created on Apr 24, 2013

@author: gena
'''
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from PIL import Image


def cumputeCodeVector(X,centroids):
    """
     The function groups elements in compliance  with given centroids
     X is [n x nFeatures] array of samples
     centroids is the [nFeatures x k] array where k is the number of centers 
    """
    nCenters = np.size(centroids,1)
    nSamples = len(X)
    
    dist = euclidianDistance(X,centroids)
    ind = np.argmin(dist,1)
    code_v = np.zeros((nSamples,nCenters))
    for i in range(nSamples):
        code_v[i,ind[i][0,0]] = 1
    
    return code_v   

def findCentroids(X,ind,K):
    """
     Finds new centers according to code vector ind
     X is [n x nFeatures] array of samples
     ind is code vector
     K is amount of clusters
    """
    centroids = np.zeros((np.size(X,1),K))
    empty_clusters = list()
    #clusters = list()
  
    for i in range(K):
        x_ = X[ind[:,i] == 1]
        len_x_ = len(x_)
        #clusters.append(len_x_)
        if len_x_ < 10: #we found small clusters
            empty_clusters.append(i)
            #print "empty clasters was found"
            continue
        
        centroids[:,i] = np.sum(x_,0)/float(len_x_)
    #print clusters  
    diff = np.setdiff1d(range(K), empty_clusters)  
    return centroids[:,diff],ind[:,diff]
    
def euclidianDistance(X,Y):
    """
     X is [n x nFeatures] array
     Y is the [nFeatures x k] array
    """
    return np.sqrt(np.mat(np.sum(np.power(X,2),1)) + np.mat(np.sum(np.power(Y,2),0)) - 2*np.dot(X,Y))

def costFunction(centroids,code,X):
    """
      Compute the reconstruction error
      centroids is [nFeatures x k] matrix
      code is {0,1} [nSamples x k] code matrix
      X is [nSamples x nFeatures]
    """
    return np.sum(np.abs(np.dot(code,centroids.T) - X))

def featureNormilize(X,additive = 10):
    mu = np.array(np.mean(X,1),ndmin = 2).T
    var = np.array(np.sum(np.power(X-mu,2),1)/float(np.size(X,1)),ndmin = 2).T
    std = np.sqrt(var + additive)
    return (X-mu)/std,mu,std  

def recoverProjection(X_transformed,U):
    return np.dot(X_transformed,U)


def runK_means(X,initial_centroids,K,num_iter):
    """ K-means clustering
    
    Parameters
    ----------
    X : array-like,shape =  (n_samples x n_features)
        Coordinates of the data points
        
    K : int 
        The number of clusters
    
    initial_centroids: array-like, shape = (n_features , n_centers] 
        Initially initialized clusters
        
    Returns
    -------
    centers: ndarray, shape = (n_features , n_remaining_clusters)
        Returns found centroids
     
    Notes
    -----
    The amount of found clusters may be different from the amount of the
    initial clusters, since some of them were were near-empty during the execution     
      
    """
    centroids = initial_centroids
    
    err = 0
    prevErr = np.inf
    sumErr = 0
    eps = 0.0001
    
    for i in range(1,num_iter+1):
        v_ind = cumputeCodeVector(X,centroids)
        
        centroids,v_ind= findCentroids(X,v_ind,np.size(centroids,1))
        err = costFunction(centroids,v_ind,X)/len(X)
        print "iter: " + str(i) + " err: " + str(err) + " K = " + str(np.size(centroids,1))
        sumErr +=err 
        if np.abs(err - prevErr) < eps: #
            break
        else:
            prevErr = err
        
    return centroids,cumputeCodeVector(X,centroids)

def whitening(X,e):
    cov = np.dot(X.T,X)/len(X)
    V,D,V_t = np.linalg.svd(cov)
    return np.dot(np.dot(X,V)/np.sqrt(D + e),V_t)
    
def initializeClusters(nfeatures,K):
    #return np.random.normal(0,1,(nfeatures,K))*np.array(std,ndmin = 2).T + np.array(mu,ndmin = 2).T   
    return np.random.normal(0,1,(nfeatures,K))

def trainMiniBatchK_means(X,initial_centroids,batch_size,num_iters):
    """Mini-Batch K-Means clustering
     
    Parameters
    ----------
    X: array-like, shape = (nSamples , nFeatures)
        Coordinates of the data points    
    
    initial_centroids: array-like, shape = (nFeatures , nCetriods)
        Initially initialized clusters
        
    batch_size: int
        The size of the batches
        
    num_iters: int
        Maximum number of iterations
        
    Returns
    -------
    centers: ndarray, shape [nFeatures x nClusters]
        Returns found centroids
     
    Notes
    -----
    The amount of found clusters may be different from the amount of the
    initial clusters, since some of them were empty during the execution     
    """
    
    centroids = initial_centroids 
    v_c = np.zeros((np.size(initial_centroids,1),)) # per-center counts
    for i in range(num_iters):
        
        ind = np.random.randint(0,len(X),batch_size)
        x_ = X[ind,:]
        v_ind = cumputeCodeVector(np.mat(x_),centroids)
        #v_c = v_c + sum(v_ind,1) #the amount of elements in each cluster
        for j in range(len(v_ind)):
            c_ind = np.where(v_ind[j,:]==1)[0][0]
            v_c[c_ind]+=1
            nu = 1/float(v_c[c_ind])
            centroids[:,c_ind] = (1 - nu)*centroids[:,c_ind] + nu*x_[j,:]
        
    not_empty_clusters = (v_c != 0)
    if not all(not_empty_clusters):
        print "found empty clusters"
        print v_c
        centroids = centroids[not_empty_clusters] 
    return centroids          
        

def visualizeFilters(W):
    #W = W - np.mean(W) #rescale
    l,m = np.shape(W)
    nCols = int(np.ceil(np.sqrt(m)))
    nRows = nCols
    diff = m - np.power(nCols,2)
    
    if diff != 0:
        if diff > 0: #we should add rows
            md = diff / nCols
            nRows = nRows + md + int((diff - md) > 0)
        else:
            md = np.abs(diff) / nCols
            nRows = nRows - md 
            
            
    pw=np.sqrt(l)
    ph = pw
    
    hs = 2
    ws = 2
    
    fig1 = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
   
    arr = np.ones((nRows*ph + (nRows + 1)*hs,nCols*pw + (nCols+1)*ws))*-1
    immap = 'gray'
       
    k = 0
    for i in range(0,nRows):
        diff = m - nCols*(i+1)
        nC = int(diff>0)*nCols + (nCols - np.abs(diff))*int(diff<0)
        for j in range(0,nC):
            #norm=np.max(np.abs(W[:,k]));
            p1 = i*ph + hs*(i+1)
            p2 = j*pw + ws*(j+1)
            arr[p1:p1+ph,p2:p2+pw] = np.reshape(W[:,k],(pw,ph))#/norm
            k+=1
    
    plt.imshow(arr, cmap=immap)
    plt.tight_layout()
    fig1.show()
 
def rescaleImage(X,from_,to):
    img = Image.fromarray(np.reshape(X,from_))
    return np.array(img.resize(to).getdata()).flatten()
    

