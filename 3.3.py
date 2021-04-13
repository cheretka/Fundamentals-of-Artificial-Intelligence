import numpy as np
from numpy import exp, array, random, dot, tanh
import pickle


class NeuralNetwork():

    def __init__(self, input_neurons, tab_hidden_neurons, output_neurons, alpha):
        self.alpha = alpha
        self.weight_matrix = []

        self.weight_matrix.append(2 * np.random.random((tab_hidden_neurons, input_neurons)) - 1)

        self.weight_matrix.append(2 * np.random.random((output_neurons, tab_hidden_neurons)) - 1)

        print(self.weight_matrix)



    def predict(self, input):
        layer_1_values = self.relu(np.dot(input, self.weight_matrix[0].T))
        layer_2_values = np.dot(layer_1_values, self.weight_matrix[1].T)

        return layer_2_values

    def fit(self, input, expected_output):

        layer_1_values = self.relu(np.dot(input, self.weight_matrix[0].T))
        layer_2_values = np.dot(layer_1_values, self.weight_matrix[1].T)

        layer_2_delta = layer_2_values - expected_output
        layer_1_delta = np.dot(layer_2_delta, self.weight_matrix[1]) * self.relu2deriv(layer_1_values)

        layer_2_weight_delta = np.outer(layer_2_delta, layer_1_values)
        layer_1_weight_delta = np.outer(layer_1_delta, input)

        self.weight_matrix[1] = self.weight_matrix[1] - np.dot(self.alpha, layer_2_weight_delta)
        self.weight_matrix[0] = self.weight_matrix[0] - np.dot(self.alpha, layer_1_weight_delta)

        # input = np.array(input)
        #
        # layer_1 = self.relu(np.dot(input, self.weight_matrix[0]))
        # layer_2 = np.dot(layer_1, self.weight_matrix[1])
        #
        # layer_2_delta = (layer_2 - expected_output)
        # layer_1_delta = layer_2_delta.dot(self.weight_matrix[1].T) * self.relu2deriv(layer_1)
        #
        # self.weight_matrix[1] -= self.alpha * layer_1.T.dot(layer_2_delta)
        # self.weight_matrix[0] -= self.alpha * input.T.dot(layer_1_delta)


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

    network = NeuralNetwork(784, 40, 1, 0.01)


    file_labels_train = open("train-labels.idx1-ubyte", "rb")
    print(int.from_bytes(file_labels_train.read(4), "big"))
    print(int.from_bytes(file_labels_train.read(4), "big"))
    print()

    file_images_train = open("train-images.idx3-ubyte", "rb")
    print(int.from_bytes(file_images_train.read(4), "big"))
    print(int.from_bytes(file_images_train.read(4), "big"))
    print(int.from_bytes(file_images_train.read(4), "big"))
    print(int.from_bytes(file_images_train.read(4), "big"))


    for i in range(6):
        images = []
        for k in range(28*28):
            images.append(int.from_bytes(file_images_train.read(1), "big"))

        # images = fun(images)
        label = int.from_bytes(file_labels_train.read(1), "big")
        # print("image")
        # print(images)
        # print(label)
        network.fit(images, label)


        if i%5000 == 0:
            print('.')

    print()
    print()





    # file_labels_test = open("train-labels.idx1-ubyte", "rb")
    # print(int.from_bytes(file_labels_test.read(4), "big"))
    # print(int.from_bytes(file_labels_test.read(4), "big"))
    # print()
    #
    # file_images_test = open("train-images.idx3-ubyte", "rb")
    # print(int.from_bytes(file_images_test.read(4), "big"))
    # print(int.from_bytes(file_images_test.read(4), "big"))
    # print(int.from_bytes(file_images_test.read(4), "big"))
    # print(int.from_bytes(file_images_test.read(4), "big"))
    # print()
    #
    # for i in range(10):
    #     images = int.from_bytes(file_images_test.read(28*28), "big")
    #     images = network.relu2deriv(images)
    #     label = int.from_bytes(file_labels_test.read(1), "big")
    #
    #     output = network.predict(images)
    #
    #     print(label)
    #     print(len(output))
    #     # print(output)
    #     print()