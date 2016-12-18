""" Impementation Of Multi_Varative Linear Regression using gradient Descent  """

import numpy as np
from numpy import genfromtxt
from statistics import stdev
import matplotlib.pyplot as plt
from math import fabs
import random

class MultiVariate():
    def __init__(self, features, values) :
        self.features = features
        self.values = values
        self.means = []
        self.stdevs = []
        self.scale_features()
        self.randomize()

                
    def scale_features(self):
        for i in range(self.features.shape[1]):
            mean = self.features[:,i].sum() / self.features.shape[0]
            s =  stdev(self.features[:,i])
            self.means.append(mean)
            self.stdevs.append(s)
            for j in range(self.features.shape[0]):
                self.features[j,i] = (self.features[j,i] - mean) / s
        self.features = np.concatenate((np.ones((self.features.shape[0], 1)), self.features), axis=1)
                
    def randomize(self):
        z = np.concatenate((self.features, self.values), axis =1)
        np.random.shuffle(z)
        self.features, self.values = z[:,:-1], z[:,-1:]

    def train(self, alpha = .1, n1 = 10, n2 = 10, iterations = 400):
        
        self.features_train = self.features[:n1,:]
        self.features_test = self.features[-n2:,:]
        self.values_train = self.values[:n1,:] 
        self.values_test = self.values[-n2:, :]
        theta = np.ones((self.features.shape[1],1))
        
        m = self.features_train.shape[0]
        for l in range(iterations):
                h = np.dot(self.features_train, theta)
                for j in range(len(theta)):
                        theta[j] = theta[j] - alpha * (1/m) * sum(np.multiply((h -self.values_train), self.features_train[:,j:j+1]))[0]
        self.theta = theta
        return [theta,self.means,self.stdevs];
            
    def plot(self,f):
        
        r = np.dot(self.features_test, self.theta)
        
        abline_values = [self.theta[f] * i + self.theta[0] for i in self.features_test[:,f] ]

        plt.close()
        plt.scatter(self.features_test[:,f:f+1],self.values_test)
        plt.plot(self.features_test[:,f:f+1], abline_values,color='r')
        plt.show()
        
    def predict(self,f):
        
        return np.dot( f, self.theta)

"""-------------------    Example code------------------------------------------------------------------------"""

#generating numpy array from tsv file

data = genfromtxt('linear_regression_data4.txt', delimiter=",")

# separating features and actual values
features, values = data[:,:-1], data[:,-1:]

clf = MultiVariate(features, values)
[theta, means, stdevs ] = clf.train(.01, 37, 10, 400) # learning rate, no of training features, no of testing features
clf.plot(1) #plot first feature

#predicting prices for a 1400 sq-ft 4 BHK house
r = clf.predict([ 1, (1400-means[0])/stdevs[0], (4-means[1])/stdevs[1]])
print("Housing Prices for a 1400 sq-ft 4 BHK house is:",r[0])
