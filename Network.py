# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:00:51 2019

@author: Altruy

"""
import numpy as np
import sys
import time
import matplotlib.pyplot as plt


class NeuralNetwork():
    @staticmethod
    # note the self argument is missing i.e. why you have to search how to use static methods/functions
    def cross_entropy_loss(y_pred, y_true):
        return -(y_true * np.log(y_pred)).sum()

    @staticmethod
    def accuracy(y_pred, y_true):
        count = 0

        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                count += 1
        return count/len(y_pred)*100

    @staticmethod
    def softmax(x):
        expx = np.exp(x)
        return expx / expx.sum(keepdims=True)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def __init__(self, files, file1):
        self.num_layers = 3
        self.nodes_per_layer = [784, 30, 10]
        self.input_shape = 784
        self.output_shape = 10
        self.lr = []
        self.t = []
        self.data = []
        self.labels = []
        self.predictions = []
        self.filen = [files, file1]
        self.__init_weights(self.nodes_per_layer)

    def __init_weights(self, nodes_per_layer):
        self.weights_ = []
        for i, _ in enumerate(nodes_per_layer):
            if i == 0:
                # skip input layer, it does not have weights/bias
                continue
            weight_matrix = np.random.normal(
                size=(nodes_per_layer[i-1], nodes_per_layer[i]))
            self.weights_.append(weight_matrix)

    def fit(self, epochs, lr=1e-3):
        history = []
        acc = []
        print('Begining fit')
        k = 1
        self.lr.append(lr)
        start = time.time()
        for _ in range(epochs):
            print("running epoch:", k)

            for i in range(len(self.data)):
                activations = self.forward_pass(i)
                self.backward_and_update(i, activations, lr)
            self.predict()
            current_loss = self.cross_entropy_loss(
                self.predictions, self.labels)
            history.append(current_loss)
            acc.append(self.evaluate()[0])
            k += 1
        end = time.time()-start
        self.t.append(end)

        print('fit with %f in %f' % (lr, end))
        self.save_weights('netWeights.txt')

        return history, acc

    def forward_pass(self, i):
        activations = []
        hidden_layer = np.dot(self.data[i], self.weights_[0])
        hidden_layer_activation = NeuralNetwork.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer_activation, self.weights_[1])
        output_layer_activation = NeuralNetwork.softmax(output_layer)
        activations.append(hidden_layer_activation)
        activations.append(output_layer_activation)
        return activations

    def backward_and_update(self, i, layer_activations, lr):
        derivative_o = layer_activations[1] - self.labels[i]
        derivative_h = layer_activations[0]

        d_sigmoid = np.dot(self.data[i], self.weights_[0])
        d_sigmoid = NeuralNetwork.sigmoid(
            d_sigmoid) * (1 - NeuralNetwork.sigmoid(d_sigmoid))
        j = d_sigmoid * np.dot(self.weights_[1], derivative_o).T

        deltas0 = np.dot(self.data[i].reshape(784, 1), j.reshape(30, 1).T)
        deltas1 = np.dot(derivative_h.reshape(30, 1),
                         derivative_o.reshape(10, 1).T)

        self.weights_[0] -= lr*deltas0
        self.weights_[1] -= lr*deltas1

    def predict(self):
        self.predictions = [self.forward_pass(
            i)[1] for i in range(len(self.data))]

    def evaluate(self):
        acc = self.accuracy(np.array(self.predictions).argmax(
            axis=1), np.array(self.labels).argmax(axis=1))
        loss = self.cross_entropy_loss(self.predictions, self.labels)
        return acc, loss

    def give_images(self):
        print('loading images')
        images = []
        labels = []
        temp = []

        f = open(self.filen[0], 'r')
        for i in f.read().split():
            if ']' in i:
                temp.append(float(i[0:-1]))
                images.append(temp)
                temp = []
            elif i == '[':
                continue
            else:
                temp.append(float(i))
        f.close()

        f = open(self.filen[1], 'r')
        labels = f.read().splitlines()
        one_hot = []
        for i in labels:
            temp = []
            for j in range(10):
                if j == int(i):
                    temp.append(1)
                else:
                    temp.append(0)
            one_hot.append(temp)
        f.close()
        labels = np.array(one_hot)

        images = (images - np.mean(images)) / np.std(images)
        self.data = images
        self.labels = labels
        print('done loading images')

    def save_weights(self, file):
        f = open(file, 'w')
        for i in self.weights_:
            for j in i:
                for k in j:
                    f.write(str(k)+'\n')
        f.close()

    def reassign_weights(self, file):
        f = open(file, 'r')

        for i in range(2):
            if i == 0:
                for j in range(self.nodes_per_layer[0]):
                    for k in range(self.nodes_per_layer[1]):
                        self.weights_[i][j][k] = float(f.readline())

            elif i == 1:
                for j in range(self.nodes_per_layer[1]):
                    for k in range(self.nodes_per_layer[2]):
                        self.weights_[i][j][k] = float(f.readline())
        f.close()

    def savePlot(self):
        plt.plot(self.lr, self.t)
        plt.xlabel('Learning Rate')
        plt.ylabel('Execution Time / (s)')
        plt.savefig('graph.pdf')
        plt.show()

    def test(self, weightfile):
        self.reassign_weights(weightfile)
        self.give_images()
        self.predict()
        acc, loss = self.evaluate()
        print('Accuracy:', acc)
        print('Loss:', loss)

    def train(self, lr):
        self.give_images()
        history = self.fit(2, lr)
        print('history:', history[0])
        print('Accuracy:', history[1])


def main():
    brain = NeuralNetwork(sys.argv[2], sys.argv[3])
    if sys.argv[1] == 'test':
        brain.test(sys.argv[4])

    elif sys.argv[1] == 'train':
        lr = float(sys.argv[4])
        brain.train(lr)

    elif sys.argv[1] == 'graph':
        lr = float(sys.argv[4])
        brain.train(lr)
        for _ in range(3):
            lr = float(input('Lr:'))
            brain.train(lr)
        brain.savePlot()


main()
