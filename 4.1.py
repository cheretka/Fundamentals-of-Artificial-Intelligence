import numpy as np
from numpy import exp, array, random, dot, tanh
import pickle
import sys

class NeuralNetwork():

    def __init__(self, input_neurons, tab_hidden_neurons, output_neurons, alpha):
        self.alpha = alpha
        # zakres <-0.1; 0.1>
        self.layer_1_weights = 0.2*np.random.random((input_neurons, tab_hidden_neurons)) - 0.1
        self.layer_2_weights = 0.2*np.random.random((tab_hidden_neurons, output_neurons)) - 0.1

        self.batch_size = tab_hidden_neurons

    def predict(self, input):
        layer_1_values = self.relu(np.dot(input, self.layer_1_weights.T))
        layer_2_values = np.dot(layer_1_values, self.layer_2_weights.T)
        return layer_2_values



    def fit(self, images, labels, iteration,  test_images, test_labels):

        for j in range(iteration):
            error, correct_cnt = (0.0000, 0)

            for i in range(len(images)):

                input = np.array(images[i:i + 1])
                layer_1_values = self.relu(np.dot(input, self.layer_1_weights))
                dropout_mask = np.random.randint(2, size=layer_1_values.shape)
                layer_1_values = layer_1_values * (dropout_mask * 2)
                layer_2_values = np.dot(layer_1_values, self.layer_2_weights)

                error += np.sum((labels[i:i + 1] - layer_2_values) ** 2)
                correct_cnt += int(np.argmax(layer_2_values) == np.argmax(labels[i:i + 1]))

                layer_2_delta = (labels[i:i+1] - layer_2_values)

                layer_1_delta = layer_2_delta.dot(self.layer_2_weights.T) * self.relu_deriv(layer_1_values)
                layer_1_delta *= dropout_mask

                self.layer_2_weights = self.layer_2_weights + self.alpha * layer_1_values.T.dot(layer_2_delta)
                self.layer_1_weights = self.layer_1_weights + self.alpha * input.T.dot(layer_1_delta)

            if j % 10 == 0:
                test_error = 0.0
                test_correct_cnt = 0

                for i in range(len(test_images)):
                    input = test_images[i:i + 1]
                    layer_1_values = self.relu(np.dot(input, self.layer_1_weights))
                    layer_2_values = np.dot(layer_1_values, self.layer_2_weights)

                    test_error += np.sum((test_labels[i:i + 1] - layer_2_values) ** 2)
                    test_correct_cnt += int(np.argmax(layer_2_values) == np.argmax(test_labels[i:i + 1]))

                sys.stdout.write("\n" + \
                                 "I:" + str(j) + \
                                 " Test-Err:" + str(test_error / float(len(test_images)))[0:5] + \
                                 " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + \
                                 " Train-Err:" + str(error / float(len(images)))[0:5] + \
                                 " Train-Acc:" + str(correct_cnt / float(len(images))))


    def relu(self, x):
        return (x > 0) * x

    def relu_deriv(self, output):
        return output > 0

    def save_weights(self, file_name):
        pickle.dump(self, open(file_name, "wb"))

    def load_weights(self, file_name):
        return pickle.load(open(file_name, "rb"))






if __name__ == "__main__":

    input_neurons = 784
    tab_hidden_neurons = 40
    output_neurons = 10
    alpha = 0.005

    batch_size = tab_hidden_neurons

    number_images_train = 1000
    number_images_test = 10000
    iteration = 350

    network = NeuralNetwork(input_neurons, tab_hidden_neurons, output_neurons, alpha)

    file_labels_train = open("train-labels.idx1-ubyte", "rb")
    print(int.from_bytes(file_labels_train.read(4), "big"))
    print(int.from_bytes(file_labels_train.read(4), "big"))
    print()

    file_images_train = open("train-images.idx3-ubyte", "rb")
    print(int.from_bytes(file_images_train.read(4), "big"))
    print(int.from_bytes(file_images_train.read(4), "big"))
    print(int.from_bytes(file_images_train.read(4), "big"))
    print(int.from_bytes(file_images_train.read(4), "big"))

    all_images = []
    all_label = []

    for i in range(number_images_train):

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



    print()
    # print("--------------------------test---------------------------")

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

    all_images_test = []
    all_label_test = []

    for i in range(number_images_test):
        images = []
        for k in range(28 * 28):
            images.append(int.from_bytes(file_images_test.read(1), "big"))

        images = np.array(images)
        images = images / 255

        label = int.from_bytes(file_labels_test.read(1), "big")
        exp = np.zeros(10)
        exp[label] = 1

        all_images_test.append(images)
        all_label_test.append(exp)



    network.fit(all_images, all_label, iteration, all_images_test, all_label_test)

