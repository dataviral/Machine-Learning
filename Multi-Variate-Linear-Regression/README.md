
<h1>MULTIVARIATE LINEAR REGRESSION</h1>

Python Implementation of MULTIVARIATE LINEAR REGRESSION Classifier
Learning Algorithm --> Gradient Descent
With Visualisation of the Fit

Requirements : 
	1. numpy
 	2. matplotlib

Class MultiVariate():

	ROUTINES :
 
	1. scale_features(self) : 
		Performs Feature Scaling to reduce skewness of the Cost_Function vs Parameter graph
		Called Internally by train

	2. randomize(self) :
		randomizes the features 
        
	3. train(self, alpha,lamb, n1, n2) :
		Performs Gradient Descent and fits our linear hypothesis model to the training set
	
		# Arguments					
			alpha -> Learning_Rate
			lamb -> Regularization Parameter
			n1    -> no of training features
			n2    -> no of testing features

	4. plot(self,f) : Plots the hypotesis pedictions of the test set along with a scatter
		     plot of the actual test set values
			# Arguments
				f -> feature to plot against 
	5. predict(self, ft) : Predict for a test-set
			# Arguments
				ft -> test-set

