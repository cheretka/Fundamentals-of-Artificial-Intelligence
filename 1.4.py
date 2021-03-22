import numpy as np


# input_values - vector [3] /   weight_matrix - macierz [3, 3]
def deep_neural_network(input_vector, weight_matrix):
    return np.dot(input_vector, weight_matrix)

print("input")
input_values = np.array([[8.5, 0.65, 1.2],
                         [9.5, 0.8, 1.3],
                         [9.9, 0.8, 0.5],
                         [9.0, 0.9, 1.0]])
print(input_values)
print()

print("layer 1 weights")
weight_matrix_1 = np.array([[0.1, 0.2, -0.1],
                            [-0.1, 0.1, 0.9],
                            [0.1, 0.4, 0.1]]).T
print(weight_matrix_1)
print()

print("layer 2 weights")
weight_matrix_2 = np.array([[0.3, 1.1, -0.3],
                            [0.1, 0.2, 0.0],
                            [0.0, 1.3, 0.1]]).T
print(weight_matrix_2)
print()

print("results")


for vector in input_values:
    output_values1 = deep_neural_network(vector, weight_matrix_1)
    print(output_values1)
    output_values2 = deep_neural_network(output_values1, weight_matrix_2)
    print(output_values2)
    print()
