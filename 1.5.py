import numpy as np
from numpy import exp, array, random, dot, tanh


class NeuralNetwork():

    def __init__(self, input_neurons, output_neurons, tab_hidden_neurons):
        self.num_input = input_neurons
        self.num_output = output_neurons

        self.n_layers = [input_neurons]
        for i in tab_hidden_neurons:
            self.n_layers.append(i)
            self.n_layers.append(i)
        self.n_layers.append(output_neurons)
        self.n_layers = np.reshape(np.array(self.n_layers), (-1, 2))
        print(self.n_layers)

        self.weight_matrix = []

        for index in self.n_layers:
            weight = (2 * random.random((index[1], index[0])) - 1)
            self.weight_matrix.append(weight)

        print(self.weight_matrix)




    def add_layer(self, n, weight_min_value=-1, weight_max_value=1):
        # del self.n_layers[-1]
        self.n_layers[:-1]
        print(self.n_layers)
        np.append(self.n_layers, [[self.n_layers[len(self.n_layers)-1][0], n]])
        print(self.n_layers)
        # self.n_layers[len(self.n_layers)-1] = [ self.n_layers[len(self.n_layers)-1][0], n]
        np.append(self.n_layers, [[n, self.num_output]])
        # self.n_layers.append([n, self.num_output])
        print(self.n_layers)


    def predict(self, input_vector):
        output = input_vector

        for mat in self.weight_matrix:
            # print(output)
            # print(mat.T)
            output = np.dot(output, mat.T)
            # print(output)

        return output







if __name__ == "__main__":

    network = NeuralNetwork(4, 3, [2, 3, 4])

    o = network.predict([1, 2, 3, 4])
    print("o")
    print(o)
    network.add_layer(7)



