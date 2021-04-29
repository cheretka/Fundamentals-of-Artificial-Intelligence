import numpy as np
from numpy import exp, array, random, dot, tanh
import pickle


class NeuralNetwork():

    def __init__(self, input_neurons, tab_hidden_neurons, output_neurons, alpha):
        self.alpha = alpha
        self.layer_1_weights = (2 * np.random.random((tab_hidden_neurons, input_neurons)) - 1) / 10
        self.layer_2_weights = (2 * np.random.random((output_neurons, tab_hidden_neurons)) - 1) / 10

    def predict(self, input):
        layer_1_values = self.sigmoid(np.dot(input, self.layer_1_weights.T))
        layer_2_values = np.dot(layer_1_values, self.layer_2_weights.T)
        return layer_2_values

    def fit(self, input, expected_output):
        layer_1_values = self.sigmoid(np.dot(input, self.layer_1_weights.T))
        layer_2_values = np.dot(layer_1_values, self.layer_2_weights.T)

        layer_2_delta = layer_2_values - expected_output
        layer_1_delta = np.dot(layer_2_delta, self.layer_2_weights) * self.sigmoid2deriv(layer_1_values)

        layer_2_weight_delta = np.outer(layer_2_delta, layer_1_values)
        layer_1_weight_delta = np.outer(layer_1_delta, input)

        self.layer_2_weights = self.layer_2_weights - np.dot(self.alpha, layer_2_weight_delta)
        self.layer_1_weights = self.layer_1_weights - np.dot(self.alpha, layer_1_weight_delta)


    # def relu(self, x):
    #     return (x > 0) * x
    #
    # def relu2deriv(self, output):
    #         return output>0

    def save_weights(self, file_name):
        pickle.dump(self, open(file_name, "wb"))

    def load_weights(self, file_name):
        return pickle.load(open(file_name, "rb"))

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def sigmoid2deriv(self, output):
        return output*(1-output)



if __name__ == "__main__":

    all_rewiews = []
    numberOfAllRewiews = 0
    with open("rewiews.txt") as fp:
        Lines = fp.readlines()
        for line in Lines:
            all_rewiews.append(line)
            numberOfAllRewiews +=1
    print(numberOfAllRewiews)

    numberOfRewiews_train = int(numberOfAllRewiews*24 /25)
    numberOfRewiews_test = int(numberOfAllRewiews /25)
    print(numberOfRewiews_train)
    print(numberOfRewiews_test)

    rewiews_train = all_rewiews[ 0:numberOfRewiews_train]
    rewiews_test = all_rewiews[numberOfRewiews_train : numberOfRewiews_train+numberOfRewiews_test]

    all_labels = []
    numberOfAllLabels = 0
    with open("labels.txt") as fp:
        Lines = fp.readlines()
        for line in Lines:
            if line == "negative\n":
                all_labels.append(0)
            else:
                all_labels.append(1)
            numberOfAllLabels +=1
    print(numberOfAllLabels)

    labels_train = all_labels[0:numberOfRewiews_train]
    labels_test = all_labels[numberOfRewiews_train: numberOfRewiews_train + numberOfRewiews_test]
    print(len(labels_train))
    print(len(labels_test))
##################################################################
    print("\n\n\n")
    unique_list = []
    licz = 0
    for rewiew in all_rewiews:
        all_words = rewiew.split()
        for word in all_words:
            if word not in unique_list:
                unique_list.append(word)
        licz += 1
        if licz % 100 == 0:
            print(licz)

    # print(unique_list)
    print("len(unique_list)=")
    print(len(unique_list))

    lenInput = len(unique_list)


######################## train ############################
    print("\n\n\n")

    network = NeuralNetwork(lenInput, 300, 1, 0.01)
    licz = 0
    for count in range(2):
        for k in range(numberOfRewiews_train):
            wektor = []
            # print(rewiews_train[k])

            for i in range(lenInput):
                if unique_list[i] in rewiews_train[k]:
                    wektor.append(1)
                else:
                    wektor.append(0)

            # print(wektor)
            network.fit(wektor, labels_train[k])
            licz+=1
            if licz%100==0:
                print(licz)


######################## test ##################################

    a_file = open("weights1.txt", "w")
    for row in network.layer_1_weights:
        np.savetxt(a_file, row)

    a_file.close()

    a_file2 = open("weights2.txt", "w")
    for row in network.layer_2_weights:
        np.savetxt(a_file2, row)

    a_file2.close()

    a_file3 = open("unique_list.txt", "w")
    for row in unique_list:
        np.savetxt(a_file3, row)

    a_file3.close()


    print(network.layer_1_weights)
    print()
    print(network.layer_2_weights)
    print()
    print(unique_list)
    print()
    print("done")


    # print("\n\n\n")
    #
    # good_points = 0
    # bad_points = 0
    # licz = 0
    # for k in range(numberOfRewiews_test):
    #     wektor = []
    #     # print(rewiews_test[k])
    #
    #     for i in range(lenInput):
    #         if unique_list[i] in rewiews_test[k]:
    #             wektor.append(1)
    #         else:
    #             wektor.append(0)
    #
    #     # print(wektor)
    #     output = round( network.predict(wektor))
    #     if output == labels_test[k]:
    #         good_points += 1
    #     else:
    #         bad_points += 1
    #
    #     licz += 1
    #     if licz % 100 == 0:
    #         print(licz)
    #
    # print("good_points")
    # print(good_points)
    # print("bad_points")
    # print(bad_points)
