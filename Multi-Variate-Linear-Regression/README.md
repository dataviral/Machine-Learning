#MULTIVARIATE LINEAR REGRESSION

Requirements : 
	1. numpy
 	2. matplotlib

Class MultiVariate():

	ROUTINES ->
 
	1. scale_features(self) : 
		Performs Feature Scaling to reduce skewness of the Cost_Function vs Parameter graph
		Called Internally by train

	2. randomize() :
		randomizes the features 
        
	3. train(self, alpha, n1, n2) :
		Performs Gradient Descent and fits our linear hypothesis model to the training set
	
		# Arguments					
			alpha -> Learning_Rate
			n1    -> no of training features
			n2    -> no of testing features

	4. plot(f) : Plots the hypotesis pedictions of the test set along with a scatter
		     plot of the actual test set values
			# Arguments
				f -> feature to plot against 

