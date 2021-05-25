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
            self.weight_matrix.append((2 * random.random((index[1], index[0])) - 1))

        print(self.weight_matrix)

    def add_layer(self, n, weight_min_value=-1, weight_max_value=1):
        self.n_layers = self.n_layers[:-1]
        if len(self.n_layers) > 0:
            self.n_layers = np.append(self.n_layers, [[self.n_layers[len(self.n_layers)-1][1], n]])
        else:
            self.n_layers = np.append(self.n_layers, [[self.num_input, n]])
        self.n_layers = np.append(self.n_layers, [[n, self.num_output]])
        self.n_layers = np.reshape(np.array(self.n_layers), (-1, 2))
        # print(self.n_layers)

        # print(self.weight_matrix)
        self.weight_matrix = self.weight_matrix[:-1]
        self.weight_matrix.append(( (abs(weight_min_value) + abs(weight_max_value)) * random.random((self.n_layers[len(self.n_layers)-2][1], self.n_layers[len(self.n_layers)-2][0])) + weight_min_value))
        self.weight_matrix.append(( 2 * random.random((self.n_layers[len(self.n_layers)-1][1], self.n_layers[len(self.n_layers)-1][0])) -1))
        # print(self.weight_matrix)

    def predict(self, input_vector):
        output = input_vector

        for mat in self.weight_matrix:
            output = np.dot(output, mat.T)

        return output








if __name__ == "__main__":

    network = NeuralNetwork(4, 3, [])

    o = network.predict([1, 2, 3, 4])
    print("output")
    print(o)
    print()
    network.add_layer(5, -20, 20)
    print(network.weight_matrix)
    o = network.predict([1, 2, 3, 4])
    print("output")
    print(o)
    print()



