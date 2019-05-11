weight = 0.5 #initial random weight
goal_pred = 0.8 #the value we are trying to predict. In a supervised example, this is the ground truth value
input = 0.5 #the input value that we receive - could be from a variety of sources
learningRate = 0.3 #how fasst/slow do we learn. Too big will lead to divergence, too small will lead to long time for convergence
epochs = 200 #the number of epochs during which we try to learn

for iteration in range(epochs):
	prediction = input * weight #the prediction based on the input value
	error = (prediction - goal_pred) ** 2 #the wrongness of the prediction as measured by squared error
	derivative = input * (prediction - goal_pred) #the derivated - which shows how the prediction is impacted by the error
	weight = weight - (learningRate * derivative) #modify the weight by the derivative to minimize the error - i.e. learn
	
	print("Error:" + str(error) + " Prediction: "+ str(prediction) + " Weight:" + str(weight))
	
	
	
#At the end of the run, we now how a single input output network which can take an input and learn the weight that will allow the input to be transformed into the output sucessfully. Remember this is a single input/output network with a single weight
