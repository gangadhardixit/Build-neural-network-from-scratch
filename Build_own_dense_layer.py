### Generate Nonlinear Data

from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
import matplotlib.pyplot as plt
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1],c=y,cmap='Accent')
plt.show()


### Class for Dense layer

class Dense_layer:
    #Initialize layer in neural network
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))


    #Forward pass Initialize layer in neural network
    def forwardpass(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.biases

X, y = spiral_data(samples=4, classes=2)    

#create a dense layer 2-inputs and 1-output
dense_layer1=Dense_layer(2,1)

#Performs dot product(weight*input)+bias
dense_layer1.forwardpass(X)




