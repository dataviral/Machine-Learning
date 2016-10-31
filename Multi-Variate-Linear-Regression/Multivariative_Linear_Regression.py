""" Impementation Of Multi_Varative Linear Regression using gradient Descent  """

import numpy as np
from numpy import genfromtxt
from statistics import stdev
import matplotlib.pyplot as plt
from math import fabs
import random
from seaborn import lmplot

class MultiVariate():
    def __init__(self, features, values) :
        self.features = features
        self.values = values
        self.randomize()
        self.scale_features()
        self.theta = np.zeros((self.features.shape[1],1))
        
    def scale_features(self):
        for i in range(self.features.shape[1]):
            mean = self.features[:,i].sum() / self.features.shape[0]
            s =  stdev(self.features[:,i])
            for j in range(self.features.shape[0]):
                self.features[j,i] = (self.features[j,i] - mean) / s
        self.features = np.concatenate((np.ones((self.features.shape[0], 1)), self.features), axis=1)
            
                
    def randomize(self):
        z = np.concatenate((self.features, self.values), axis =1)
        np.random.shuffle(z)
        self.features, self.values = z[:,:-1], z[:,-1:]

    def train(self,alpha,n1=features.shape[0],n2=10):
        
        self.features_train = self.features[:n1,:]
        self.features_test = self.features[-n2:,:]
        self.values_train = self.values[:n1,:] 
        self.values_test = self.values[-n2:, :]
        
        m = self.features_train.shape[0]
        for i in range(self.features_train.shape[0]):
            row = self.features_train[i,:]
            h = np.dot(self.features_train, self.theta)
            for j in range(len(self.theta)):
                self.theta[j] = self.theta[j] - ( alpha * (1/m) * sum(h -self.values_train)* row[j] )
    
    def plot(self,f):
        print(self.theta)
        r = np.dot(self.features_test, self.theta)
        
        slope, intercept = np.polyfit(self.features_train[:,f],self.values_train, 1)
        abline_values = [slope * i + intercept for i in self.features_test[:,f] ]

        plt.close()
        plt.scatter(self.features_test[:,f:f+1],self.values_test)
        plt.plot(self.features_test[:,f:f+1], abline_values,color='r')
        plt.show()

"""-------------------    Example code------------------------------------------------------------------------"""

#generating numpy array from tsv file
data = genfromtxt('linear_regression_data4.txt', delimiter=",")

# separating features and actual values
features, values = data[:,:-1], data[:,-1:]

clf = MultiVariate(features, values)
clf.train(.1,n1= 10, n2=17) # learning rate, no of training features, no of testing features
clf.plot(1)
        


