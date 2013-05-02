'''
Created on Apr 21, 2013

@author: gena
'''
import numpy as np
from autoencoder import *

nFeatures = 784
hidden_layer_size =529
nClasses = 10

W = np.array(np.random.rand((nFeatures+1)*hidden_layer_size + (hidden_layer_size+1)*nClasses))/100
w1 = np.reshape(W[0:(nFeatures+1)*hidden_layer_size],\
                        ((nFeatures+1),hidden_layer_size))

#np.savetxt('test.txt', W)
visualizeNet(w1[1:,:])