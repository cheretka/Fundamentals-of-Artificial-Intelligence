import numpy as np
from numpy import exp, array, random, dot, tanh
import pickle


class NeuralNetwork():

    def __init__(self, input_neurons, tab_hidden_neurons, output_neurons, alpha):
        self.alpha = alpha
        self.layer_1_weights = (2 * np.random.random((tab_hidden_neurons, input_neurons)) - 1 ) / 10
        self.layer_2_weights = (2 * np.random.random((output_neurons, tab_hidden_neurons)) - 1 ) / 10

    def predict(self, input):
        layer_1_values = self.relu(np.dot(input, self.layer_1_weights.T))
        layer_2_values = np.dot(layer_1_values, self.layer_2_weights.T)
        return layer_2_values

    def fit(self, input, expected_output):

        layer_1_values = self.relu(np.dot(input, self.layer_1_weights.T))

        layer_2_delta = np.dot(layer_1_values, self.layer_2_weights.T) - expected_output

        self.layer_2_weights = self.layer_2_weights - np.dot(self.alpha, (np.outer(layer_2_delta, layer_1_values)))
        self.layer_1_weights = self.layer_1_weights - np.dot(self.alpha, (np.outer((np.dot(layer_2_delta, self.layer_2_weights) * self.relu2deriv(layer_1_values)), input)))

        # layer_1_values = self.relu(np.dot(input, self.layer_1_weights.T))
        # layer_2_values = np.dot(layer_1_values, self.layer_2_weights.T)
        #
        # layer_2_delta = layer_2_values - expected_output
        # layer_1_delta = np.dot(layer_2_delta, self.layer_2_weights) * self.relu2deriv(layer_1_values)
        #
        # layer_2_weight_delta = np.outer(layer_2_delta, layer_1_values)
        # layer_1_weight_delta = np.outer(layer_1_delta, input)
        #
        # self.layer_2_weights = self.layer_2_weights - np.dot(self.alpha, layer_2_weight_delta)
        # self.layer_1_weights = self.layer_1_weights - np.dot(self.alpha, layer_1_weight_delta)

    def relu(self, x):
        return (x > 0) * x

    def relu2deriv(self, output):
            return output>0

    def save_weights(self, file_name):
        pickle.dump(self, open(file_name, "wb"))

    def load_weights(self, file_name):
        return pickle.load(open(file_name, "rb"))






if __name__ == "__main__":

    network = NeuralNetwork(784, 40, 10, 0.01)


    file_labels_train = open("../train-labels.idx1-ubyte", "rb")
    print(int.from_bytes(file_labels_train.read(4), "big"))
    print(int.from_bytes(file_labels_train.read(4), "big"))
    print()

    file_images_train = open("../train-images.idx3-ubyte", "rb")
    print(int.from_bytes(file_images_train.read(4), "big"))
    print(int.from_bytes(file_images_train.read(4), "big"))
    print(int.from_bytes(file_images_train.read(4), "big"))
    print(int.from_bytes(file_images_train.read(4), "big"))

    all_images = []
    all_label = []

    for i in range(60000):

        images = []
        for k in range(28 * 28):
            images.append(int.from_bytes(file_images_train.read(1), "big"))

        images = np.array(images)
        images = images / 255

        label = int.from_bytes(file_labels_train.read(1), "big")
        exp = np.zeros(10)
        exp[label] = 1

        all_images.append(images)
        all_label.append(exp)

        if i % 5000 == 0:
            print('.')


    for ind in range(330):
        print(ind)
        for i in range(60000):

            network.fit(all_images[i], all_label[i])

            # if i % 5000 == 0:
            #     print('.')

    print()
    print("--------------------------test---------------------------")





    file_labels_test = open("../t10k-labels.idx1-ubyte", "rb")
    print(int.from_bytes(file_labels_test.read(4), "big"))
    print(int.from_bytes(file_labels_test.read(4), "big"))
    print()

    file_images_test = open("../t10k-images.idx3-ubyte", "rb")
    print(int.from_bytes(file_images_test.read(4), "big"))
    print(int.from_bytes(file_images_test.read(4), "big"))
    print(int.from_bytes(file_images_test.read(4), "big"))
    print(int.from_bytes(file_images_test.read(4), "big"))
    print()

    prawda = 0
    nieprawda = 0

    for i in range(10000):
        images = []
        for k in range(28 * 28):
            images.append(int.from_bytes(file_images_test.read(1), "big"))

        images = np.array(images)
        images = images / 255

        label = int.from_bytes(file_labels_test.read(1), "big")
        exp = np.zeros(10)
        exp[label] = 1

        output = network.predict(images)

        if label == (np.argmax(output)):
            prawda += 1
        else:
            nieprawda += 1

        # print()

    print("+  " + str(prawda))
    print("-  " + str(nieprawda))