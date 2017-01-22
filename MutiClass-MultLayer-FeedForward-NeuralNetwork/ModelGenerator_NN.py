class ModelGenerator:
    
    def __init__(self, network):
	import numpy as np
        self.layers = len(network)
        self.network = network
        self.weights = []
        self._generateWeights()
        
    def _generateWeights(self):
        for i in range(self.layers-1):
            y = np.random.randn(self.network[i+1], self.network[i]+1) *2 / (self.network[i]+1)**0.5
            self.weights.append(y)
                    
    def getWeightMatrices(self):
        return deepcopy(self.weights)
            
    def displayWeights(self):
        for i in range(len(self.weights)):
            print("Weights for layer ", i, "->",i+1 ," :\n\n",self.weights[i]," \n-----------------------------------------------------------------")
    pass
