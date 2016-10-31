class Predictor:
    
    def __init__(self, classes, features, data):
        
        self.classes = classes
        self.features = features
        self.data = data  
        self._data_mean = {}
        self._data_error = {}
        self.no_of_data_sets = {}
        self._acc = {}
        self.calcNoOfDataSets();
        self.dataMeanCalc()
        self.data_set_x = None
        print(self.no_of_data_sets)
        
    def calcNoOfDataSets(self):
        for cl in self.classes:
            self.no_of_data_sets[cl] = len(self.data[cl][self.features[0]])
        
    def dataMeanCalc(self):
        for clss in self.classes:
            f = {}
            a = dict(zip(self.features,[sorted(self.data[clss][feature]) for feature in self.features]))
        
            for feature in self.features:
                f[feature] = (sum(self.data[clss][feature])/len(self.data[clss][feature]) + (a[feature][int(self.no_of_data_sets[clss]/2)] + a[feature][int((self.no_of_data_sets[clss]/2) -1)])/2)/2
            self._data_mean[clss] = f
            
    def classPredictor(self):
                       
        print("Enter The Features :", self.features )
        user_features = dict(zip((self.features), [ float(input()) for i in range(len(self.features)) ]))

        for clss in self.classes:
            f=0
            for feature in self.features:
                f = f + abs(self._data_mean[clss][feature]- user_features[feature])
            self._data_error[clss] = f
        print("The Predicted Class is -> ", min(self._data_error, key = self._data_error.get))
        
                    
    def predictorAccuracy(self):
        for cl in self.classes:
            l = 0
            _error = dict(zip(self.classes, [[] for i in range(len(self.classes))]))
            for i in range(self.no_of_data_sets[cl]):
                for j in range(len(self.classes)):
                    d = 0
                    for feature in self.features:
                        d += abs(self.data[cl][feature][i] - self._data_mean[self.classes[j]][feature])
                    _error[self.classes[j]].append(d)
            acc = dict(zip(self.classes, [0 for i in range(len(self.classes))]))
            for i in range(self.no_of_data_sets[cl]):
                x = {}
                for clss in self.classes:
                    x[clss] = _error[clss][i]
                acc[min(x, key = x.get)] += 1
                #print(acc)
            self._acc[cl] = acc[cl]
        self.printDataPredictAcc()
    
    def printDataPredictAcc(self):
        y=0
        for cl in self.classes:
            x = self._acc[cl]/self.no_of_data_sets[cl]*100
            y += x
            print("\nPrediction Accuracy for ->", cl, "<- is :", x ,"%")
        print("\t\t\nTotal Acuuracy = ",y/len(self.classes)," %\n-----------------------------------------------------")
        
"""-------------------------------------------------------------Example Code----------------------------------------------------------------------------------------"        
import csv
import copy

# Creating array of dictionary of dictionary 
file = open("IDS-T01-iris.csv")
reader = csv.reader(file)
c_1 = ["virginica","versicolor","setosa"]
f_1 = ["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"]

x = dict(zip(f_1,[[] for i in f_1]))
data = {cl:copy.deepcopy(x) for cl in c_1}

#data = { "virginica" : {"Sepal.Length":[],"Sepal.Width":[],"Petal.Length":[],"Petal.Width":[]} ,"versicolor" : {"Sepal.Length":[],"Sepal.Width":[],"Petal.Length":[],"Petal.Width":[]}, "setosa" : {"Sepal.Length":[],"Sepal.Width":[],"Petal.Length":[],"Petal.Width":[]} }
next(reader)
i=0;
for row in reader:
        if(i>=0 and i <= 14):
            
            data[row[4]]["Sepal.Length"].append(float(row[0]))
            data[row[4]]["Sepal.Width"].append(float(row[1]))
            data[row[4]]["Petal.Length"].append(float(row[2]))
            data[row[4]]["Petal.Width"].append(float(row[3]))
        i += 1

   
iris_data = Predictor(c_1,f_1,data)
iris_data.predictorAccuracy()
