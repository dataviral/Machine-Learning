import math
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import itertools

def _sigmoid(theta_x):
        return 1/( 1 + pow(math.e, -theta_x) )

class LogisticRegression():
    
    def __init__(self,class_labels):
        self.labels = class_labels
        self.features_train = None
        self.features_test = None
        self.labels_train = None
        self.labels_test = None
    
    def _sigmoid(self, theta_x):
        return 1/( 1 + pow(math.e, -theta_x) )
    
    def _randomize(self, z):
        np.random.shuffle(z)
        
    def _calcNoOfDataSets(self):
            for i in range(self.labels_test.shape[0]):
                if( "testing_sets" not in self.labels[self.labels_test[i][0]].keys() ):
                      self.labels[self.labels_test[i][0]]["testing_sets"] = 1
                else:
                    self.labels[self.labels_test[i][0]]["testing_sets"] += 1
                
        
    def train(self, data, n1,n2, alpha = 1, iterations = 400,lamb=1):
        
        self._randomize(data)
        features ,labels = np.concatenate( (np.ones( (data[:,:-1].shape[0],1)), data[:,:-1]), axis = 1),data[:,-1:]
        self.features_train,self.features_test = features[:n1,:], features[-n2:,:]
        self.labels_train,self.labels_test = labels[:n1,:], labels[-n2:,:]
        no_of_classifiers = 1 if len(class_labels) == 2 else len(class_labels)
        self._calcNoOfDataSets()
        
        
        for c in range(1, no_of_classifiers+1): 
            theta = np.zeros((self.features_train.shape[1],1))
            labels_train_class = np.array([1 if i==c else 0 for i in self.labels_train]).reshape(self.labels_train.shape[0],1)
            
            for i in range(iterations):
                hyp = self._sigmoid(np.dot(self.features_train,theta))
                diff = hyp - labels_train_class
                for j in range(len(theta)):
                    if(j == 0):
                        theta[j] -= alpha * sum(diff*self.features_train[:,j:j+1]) / self.features_train.shape[0]
                    else :
                        theta[j] = theta[j]*(1- alpha*lamb/self.features_train.shape[0]) - alpha * sum(diff*self.features_train[:,j:j+1]) / self.features_train.shape[0]  
            self.labels[c]["prams"] = deepcopy(theta)
            del theta
            if len(class_labels) == 2:
                self.labels[c+1]["prams"] = deepcopy(theta)
                del theta
        
            
    def getClassPrams(self):
        return deepcopy(self.labels)
    
    def checkAccuracy(self):
        print("---------------Accuracy-------------------------------")
        for c in range(1,len(self.labels)+1):
            hyp = self._sigmoid(np.dot(self.features_test,self.labels[c]["prams"]))
            acc = 0
            fl = 0
            for i in range(hyp.shape[0]):
                if self.labels_test[i] == c and hyp[i] >= 0.5:
                    acc +=1
                if self.labels_test[i] != c and hyp[i] >= 0.9:
                    fl +=1
            print("class : ",self.labels[c]["class"],"\n\n\t Accuracy: ",acc/self.labels[c]["testing_sets"]*100,"% (on ",acc,"/",self.labels[c]["testing_sets"]," data sets)")
            print("\tFalse Positives :",fl,"\n")
        print("---------------------------------------------------------\n")
        
    
"""  ------------------------- Sample Code ----------------------------------- """

data = np.genfromtxt('IDS-T01-iris.csv', delimiter=",")
features, labels = data[:,:-1],data[:,-1:]

labels = np.array([1 if i==0 else 2 for i in labels]).reshape(labels.shape[0],1)

class_labels = {1:{"class":"setosa"},2:{"class":"versicolor"},3:{"class":"virginica"}}
clss = LogisticRegression(class_labels)
clss.train(data, 100,50,alpha=.22,lamb=0.5, iterations=50)

clss.checkAccuracy()


classPrams = clss.getClassPrams()
print(classPrams)



