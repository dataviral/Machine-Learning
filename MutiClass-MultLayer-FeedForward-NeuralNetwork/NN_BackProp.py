class BackProp():


    def __init__(self):

        self.weights = None
        self.model = None
        
    def _sigmoid(self,z):
        return 1 / ( 1 + np.exp(-z) )
    
    def scale_features(self, features, ext = False):
        if ext == False :
            means = []
            stdevs = []
            for i in range(features.shape[1]):
                mean = np.mean(features[:,i])
                s =  np.std(features[:,i])
                means.append(mean)
                stdevs.append(s)
                features[:,i] = (features[:,i] - mean) / s
            self.feature_train_data = [means, stdevs]
        else:
            for i in range(features.shape[1]):
                features[:,i] = (features[:,i] - self.feature_train_data[0][i]) / self.feature_train_data[1][i]
    
            
    def getScaledFeatureData(self):
        return deepcopy(self.feature_train_data)

    def getTrainedPrams(self):
        return deepcopy(self.weights)

    def _randomize(self,data):
        np.random.shuffle(data)

    def predictClass(self,features):
        activations = []
        #print(features.shape)
        activations.append(np.array(features).reshape(self.model[0]+1,1))

        for j in range(1,len(self.model)-1):

            activations.append( np.concatenate(
                    (np.ones((1,1)), self._sigmoid(np.dot(self.weights[j-1],
                                                                activations[j-1])
                                                                        )),axis =0 ))
        activations.append(self._sigmoid(np.dot( self.weights[-1], activations[-1] )))

        return activations[-1]


    def getNetworkDetails(self):
        network = {}

        for i in range(1, self.model[-1] + 1):

            if "class" in network.keys():
                network["class"][i] = { "no_of_training_sets":self.no_of_training_sets[i], "no_of_testing_sets":self.no_testing_sets[i] }

            else:
                network["class"] = {i: {"no_of_training_sets":self.no_of_training_sets[i], "no_of_testing_sets":self.no_testing_sets[i]} }

        for i in range(self.no_of_layer-1):

            if "weights" in network.keys():
                network["weights"][i] = self.weights[i]

            else:
                network["weights"] =  {i:self.weights[i]}

        return network

    def accuracy(self, n2):

        label = {}
        self.no_testing_sets = {x:0 for x in range(1, self.model[-1] + 1)}

        for i in range(n2):

            activations = []
            activations.append(self.features_test[i])

            for j in range(1, self.no_of_layer-1):

                    activations.append( np.concatenate(
                                            ([1], self._sigmoid(np.dot(self.weights[j-1],
                                                                            activations[j-1])
                                                                    )),axis =0 ))

            activations.append(self._sigmoid(np.dot( self.weights[-1], activations[-1] )))

            for j in range( self.model[-1]):

                if self.labels_test[i] - 1  == j :

                    self.no_testing_sets[j+1] += 1

                    if activations[-1][j] == max(activations[-1]) :

                        if(int(sum(activations[-1])) != self.model[-1]):

                            if(j+1 in label.keys()):
                                label[j+1] += 1

                            else:
                                label[j+1] = 1
                        else:
                            print("The Model Is Overfitting!  Aborting.....")
        return label

    def train( self, model, data, n1, epochs=400 , l_rate=.5, alpha=0, lamb=0, show_err=False ):

        self.model = model
        self._randomize(data)
        features, labels = data[:,:-1], data[:,-1:]
        features = np.concatenate((np.ones((features.shape[0],1)), features), axis=1)
        features_train, self.features_test = features[:n1,:],features[n1:,:]
        labels_train, self.labels_test = labels[:n1,:], labels[n1:,:]
        self.weights = ModelGenerator(self.model).getWeightMatrices()
        self.no_of_layer = len(self.model)


        no_of_training_sets = {x:0 for x in range(1,self.model[-1]+1) }


        for l in range(epochs):


            error = 0
            DEL_Prev = 0

            for f in range(features_train.shape[0]):

            # FEED FORWARD
                activations = []

                # FIRST ACTIVATION LAYER
                activations.append( features_train[f].reshape( self.model[0]+1, 1) )

                # MIDDLE ACTIVATION LAYER
                for j in range(1, self.no_of_layer-1):

                    activations.append( np.concatenate(
                                            (np.ones((1,1)),
                                                     self._sigmoid(np.dot(self.weights[j-1][:],
                                                                                    activations[j-1][:])
                                                                )),axis =0 ))
                # FINAL ACTIVATION LAYER
                activations.append(self._sigmoid(np.dot( self.weights[-1], activations[-1] )))

            #BACK PROPOGATION

                # CREATING RERUIQRED OUPUT ARRAY
                y = []

                for j in range( 1, model[-1]+1 ):

                    if labels_train[f][0] == j:
                        if l == 0:
                            no_of_training_sets[j] += 1
                        y.append(1)

                    else:
                        y.append(0)

                y = np.array(y).reshape(activations[-1].shape[0],1)

                # ERROR IN HYPOTHESIS
                error += sum((y - activations[-1])**2)

                # COMPUTING DELTAS
                deltas = []

                # LAST DELTA VALUE
                deltas.append( activations[-1] - y  )

                # MIDDLE AND FIRST DELTA VALUE
                for j in range(self.no_of_layer - 1, 1, -1):

                        if j == self.no_of_layer - 1 :
                            deltas.append( np.dot( self.weights[j-1].transpose(),deltas[-1] ) *
                                                                                  activations[j-1] *
                                                                                  ( 1- activations[j-1]  )
                                             )

                        else :
                            deltas.append( np.dot(self.weights[j-1].transpose(), deltas[-1][1:,:] )*
                                                                                  activations[j-1]  *
                                                                                        ( 1- activations[j-1] )
                                            )

                # REVERSING DELTA(S) ARRAY
                deltas.reverse()

                # CHANGE IN GRAD.

                DEL = []

                for i in range(0, self.no_of_layer-1):

                    if i == len(deltas) - 1:
                        DEL.append( np.dot( deltas[i], activations[i].transpose() ) )

                    else:
                        DEL.append( np.dot( deltas[i][1:,:], activations[i].transpose() ) )

            # WEIGHTS UPDATION

                for z in range( self.no_of_layer-1 ):


                    if z == 0:
                        self.weights[z] -= ( l_rate * DEL[z]   )

                    else:
                        self.weights[z] = (1- l_rate*lamb/features_train.shape[0]) * self.weights[z] - ( l_rate * DEL[z] ) + ( alpha * DEL_Prev[z] if DEL_Prev != 0 else DEL_Prev)

                DEL_Prev = DEL

            if show_err:
                print(error,l_rate)



        self.no_of_training_sets = no_of_training_sets


    def saveWeights(self,file):
        save_dict = {}
        save_dict["weights"] = self.weights
        with open('weights/'+ file + '.pkl', 'wb') as f:
            pickle.dump(save_dict, f)
    
    def loadWeights(self,file):
        load_dict = pickle.load(f)
        with open('weights/'+ file + '.pkl', 'rb') as f:
            self.weights = load_dict["weights"]

