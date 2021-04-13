import numpy as np
from numpy import exp, array, random, dot, tanh
import pickle


class NeuralNetwork():

    def __init__(self, input_neurons, tab_hidden_neurons, output_neurons, alpha):
        self.alpha = alpha

        self.layer_1_weights = 2 * np.random.random((tab_hidden_neurons, input_neurons)) - 1
        self.layer_2_weights = 2 * np.random.random((output_neurons, tab_hidden_neurons)) - 1

        print("111111")
        print(len(self.layer_1_weights))
        print(len(self.layer_1_weights[0]))
        print("222222")
        print(len(self.layer_2_weights))
        print(len(self.layer_2_weights[0]))


    def predict(self, input):

        layer_1_values = self.relu(np.dot(input, self.layer_1_weights.T))
        layer_2_values = np.dot(layer_1_values, self.layer_2_weights.T)

        # print("--1--")
        # print(len(layer_1_values))
        # print(len(layer_1_values[0]))
        # print("--2--")
        # print(len(layer_2_values))
        # print(len(layer_2_values[0]))

        return layer_2_values


    def fit(self, input, expected_output):

        layer_1_values = self.relu(np.dot(input, self.layer_1_weights.T))
        layer_2_values = np.dot(layer_1_values, self.layer_2_weights.T)

        # print("##1")
        # print(layer_1_values)
        # print( len(layer_1_values) )
        # # print( len(layer_1_values[0]) )
        # print("##2")
        # print(layer_2_values)
        # print(len(layer_2_values))
        # # print(len(layer_2_values[0]))

        layer_2_delta = layer_2_values - expected_output
        layer_1_delta = np.dot(layer_2_delta, self.layer_2_weights) * self.relu2deriv(layer_1_values)

        layer_2_weight_delta = np.outer(layer_2_delta, layer_1_values)
        layer_1_weight_delta = np.outer(layer_1_delta, input)

        self.layer_2_weights = self.layer_2_weights - np.dot(self.alpha, layer_2_weight_delta)
        self.layer_1_weights = self.layer_1_weights - np.dot(self.alpha, layer_1_weight_delta)




    def relu(self, x):
        return (x > 0) * x

    def relu2deriv(self, output):
            return output>0

    def save_weights(self, file_name):
        pickle.dump(self, open(file_name, "wb"))

    def load_weights(self, file_name):
        return pickle.load(open(file_name, "rb"))






if __name__ == "__main__":
    # network = NeuralNetwork(2, 5, 1, 0.01)
    #
    # for i in range(50):
    #  network.fit([1, 2], 5)
    #
    # print(network.predict([1, 2]))

    network = NeuralNetwork(784, 40, 10, 0.01)


    file_labels_train = open("train-labels.idx1-ubyte", "rb")
    print(int.from_bytes(file_labels_train.read(4), "big"))
    print(int.from_bytes(file_labels_train.read(4), "big"))
    print()

    file_images_train = open("train-images.idx3-ubyte", "rb")
    print(int.from_bytes(file_images_train.read(4), "big"))
    print(int.from_bytes(file_images_train.read(4), "big"))
    print(int.from_bytes(file_images_train.read(4), "big"))
    print(int.from_bytes(file_images_train.read(4), "big"))


    for i in range(200):
        images = []
        for k in range(28*28):
            images.append(int.from_bytes(file_images_train.read(1), "big"))

        images = np.array(images)
        images = 0 < images
        images = images.astype(int)

        label = int.from_bytes(file_labels_train.read(1), "big")
        exp = np.zeros(10)
        exp[label] = 1

        # print("images")
        # print(images)
        # print("label")
        # print(exp)

        network.fit(images, exp)

        if i%5000 == 0:
            print('.')

    print()
    print("--------------------------test---------------------------")





    file_labels_test = open("t10k-labels.idx1-ubyte", "rb")
    print(int.from_bytes(file_labels_test.read(4), "big"))
    print(int.from_bytes(file_labels_test.read(4), "big"))
    print()

    file_images_test = open("t10k-images.idx3-ubyte", "rb")
    print(int.from_bytes(file_images_test.read(4), "big"))
    print(int.from_bytes(file_images_test.read(4), "big"))
    print(int.from_bytes(file_images_test.read(4), "big"))
    print(int.from_bytes(file_images_test.read(4), "big"))
    print()

    prawda = 0
    nieprawda = 0

    for i in range(2):
        images = []
        for k in range(28 * 28):
            images.append(int.from_bytes(file_images_test.read(1), "big"))

        images = np.array(images)
        images = 0 < images
        images.astype(int)

        label = int.from_bytes(file_labels_test.read(1), "big")
        exp = np.zeros(10)
        exp[label] = 1

        output = network.predict(images)

        print(label)
        print(output)

        if (exp == output).all():
            prawda += 1
        else:
            nieprawda += 1

        # print(output)
        print()

    print("+  " + str(prawda))
    print("-  " + str(nieprawda))