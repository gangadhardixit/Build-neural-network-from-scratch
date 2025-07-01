### Generate Nonlinear Data

from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
import matplotlib.pyplot as plt
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1],c=y,cmap='Accent')
plt.show()


#Class for Dense layer 
class Dense_layer:
    #Initialize layer in neural network
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))


    #Forward pass Initialize layer in neural network
    def forwardpass(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.biases

X, y = spiral_data(samples=100, classes=3)    






#Create a Relu function within class 
class Activation_Relu:
    def forwardpass(self,inputs):
        #Output values to calculate y=max(0,input)
        self.output=np.maximum(0,inputs)


#create a dataset with 1000 records and 2 classes 
X,y=spiral_data(samples=1000,classes=3)

#create a layr with 2 input neurons & 3 output neurons
dense_layer1=Dense_layer(2,3)

#create a Relu activation object 
activation1=Activation_Relu()

#Make a forward pass
dense_layer1.forwardpass(X)

#Take output from previous layer and pass through Relu activation
activation1.forwardpass(dense_layer1.output)

print(activation1.output[:5])