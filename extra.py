import os, h5py
import numpy as  np
from scipy.stats import multivariate_normal

def recognize(Xtrain, Xdev, Xtest):
    '''Use any technique you like to train a recognizer with the training set,
    and then return the correct class labels for the test set.
    Extra credit points are provided for beating various thresholds above 33% accuracy.

    Inputs:
    Xtrain (dict of lists of (nframes,nceps) arrays):
        Xtrain[y][n][t,:] = t'th frame of n'th training utterance of word Y=y
    Xdev (dict of lists of (nframes,nceps) arrays):
        Xdev[y][n][t,:] = t'th frame of n'th devtest utterance of word Y=y
    Xtest (list of (nframes,nceps)  arrays):
        Xtest[n][t,:] = t'th frame of the n'th test utterance

    Returns:
    Y_hat (list of scalar strings): 
        Predicted label of each of the test utterances

    Implementation Warning:
    For the hidden dataset, the word labels will not be ['1','2','3'].
    Instead of hardcoding ['1','2','3'], use Xtrain.keys() or Xdev.keys().
    '''
    print(len(Xtest))
    Y_hat = []
    for x in range(len(Xtest)):
        Y_hat.append(2)
        
    return Y_hat
