# Digit-Recognition
A multi  layered perceptron learning Neural Network which can recognize 0-9 digits


The first thing to do is extract the txt files and keep the files in the same directory as the NeuralNetwork
The text files are pngs of digits in text format where each image is represented as a matrice of 28 x 28 where each value is the value of a pixel

The code has Three modes:
To run either mode you need numpy and matplotlib.pyplot

### Test mode
This mode uses the pre trained weights saved in the netWeights.txt file and runs on the test data set in the test.txt file and then checks its accuracy by comparing the results from the perceptron model with the actual labels

to run this:
```
python3 NeuralNetwork test test.txt test-labels.txt netWeights.txt
```

### Train Mode
This mode initallises the weights as random values and then trains the model over two epochs using the learning rate given , saves the final weights in the netWeights.txt file.

to run this: (Learning rate is given as 0.2 as an example)
```
python3 NeuralNetwork train train.txt train-labels.txt 0.2
```

### Graph Mode
This mode initallises the weights as random values and then trains the model over two epochs 3 times using the learning rate take as input in each run , saves the final weights in the netWeights.txt files each time and then plots the graph with the 3 learning rates given.
Note: Learning rates need to be given in ascending order

to run this:
```
python3 NeuralNetwork graph train.txt train-labels.txt
```
