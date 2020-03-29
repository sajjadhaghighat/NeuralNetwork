# In the name of God
# Sajjad Haghighat

import numpy as np
import pandas as pd


class MLP(object):
     def __init__(self, data, layers_size , act_func , etha = 0.1  , rand_scale = 0.1):
        self.dataset = data
        self.layers_size = layers_size
        self.etha = etha
        self.rand_scale = rand_scale
        self.neti = [None] * (len(layers_size)-1)
        self.oi = [None] * (len(layers_size) - 1)
        self.delta = [None] * (len(layers_size) - 1)
        self.act_func = self.sigmoid if act_func=="sigmoid" else self.relu
        self.der_func = self.sigmoid_derivative if act_func == "sigmoid" else self.relu_derivative
        self.mean_squared_error = []
        self.create_weights()

     def create_weights(self):
         nl = self.layers_size
         self.wi = [None] * (len(nl) - 1)
         for i in range(len(self.wi)):
             self.wi[i] = np.random.normal(0, self.rand_scale, (nl[i+1], nl[i]+1))
         self.wi_new = self.wi

     def train(self , epoches=1):

         for epoch in range(epoches):
             for row in self.dataset:
                Y = row[0]
                X = np.append(row[1:] / 255, 1)
                network , expected = self.feedforward(X, Y)
                self.backpropagation(network, expected)
                exit()

     def feedforward(self, X, Y):
         # Convert To 2D Array for compute Transpose
         X = np.array(X , ndmin=2)
         nl = len(self.layers_size)
         for i in range(nl-1):
            net = np.dot(self.wi[i], X.T)
            self.neti[i] = net[:, 0]
            tmp = self.act_func(net)
            if i == nl-2:
                self.oi[i] = tmp[:, 0]
            else:
                self.oi[i] = np.append(tmp, 1)#bias
                X = np.array(self.oi[i], ndmin=2)

         Y = self.squared_error(self.oi[-1],Y)
         return self.oi[-1] , Y


     def backpropagation(self , network , expected):
         print(self.neti[-1])
         exit()
         for i in range(len(self.layers_size)-2,-1,-1):
             if i == len(self.layers_size)-2:
                 # last layer
                 self.delta[i] = -2*(expected - network) * self.der_func(self.neti[i])


     def squared_error(self, o, d):
         tmp = np.zeros(10)
         tmp[d] = 1
         d = tmp
         self.mean_squared_error = np.append(self.mean_squared_error,((d - o) ** 2).mean())
         return d

     def mse(self):
         print(self.mean_squared_error.shape)
         print(self.mean_squared_error)
         print(self.mean_squared_error.mean())

     def sigmoid(self, z):
         result = 1.0 / (1.0 + np.exp(-z))
         return result

     def relu(self, z):

         if np.isscalar(z):
             result = np.max((z, 0))
         else:
             zero_aux = np.zeros(z.shape)
             meta_z = np.stack((z, zero_aux), axis=-1)
             result = np.max(meta_z, axis=-1)
         return result

     def sigmoid_derivative(self, z):

         result = self.sigmoid(z) * (1 - self.sigmoid(z))
         return result

     def relu_derivative(self, z):

         result = 1 * (z > 0)
         return result

     def softmax(x):
         e_x = np.exp(x - np.max(x))
         return e_x / e_x.sum(axis=0)

     def cross_entropy(predictions, targets, epsilon=1e-12):
         predictions = np.clip(predictions, epsilon, 1. - epsilon)
         N = predictions.shape[0]
         ce = -np.sum(targets * np.log(predictions + 1e-9))
         return ce

if __name__ == '__main__':
    X = pd.read_csv("mnist_train.csv")
    print(X.head(2))
    X = X.to_numpy()

    mlp = MLP(
        X,
        [784, 50, 32, 10],
        "sigmoid"
    )
    mlp.train(2)
    mlp.mse()
