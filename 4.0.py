import numpy as np


def relu(x):
    return (x > 0) * x


def relu_deriv(output):
    return output > 0

#-------------------------DANE------------------------

input = np.array([[8.5, 0.65, 1.2],
                  [9.5, 0.8, 1.3],
                  [9.9, 0.8, 0.5],
                  [9.0, 0.9, 1.0]])

expected_output = np.array([[0.1, 1, 0.1],
                            [0, 1, 0],
                            [0, 0, 0.1],
                            [0.1, 1, 0.2]])

alpha = 0.01

layer_1_weights = np.array([[0.1, 0.2, -0.1],
                            [-0.1, 0.1, 0.9],
                            [0.1, 0.4, 0.1]])

layer_2_weights = np.array([[0.3, 1.1, -0.3],
                            [0.1, 0.2, 0.0],
                            [0.0, 1.3, 0.1]])

#-------------------------------------------------------

layer_1_values = np.dot(input, layer_1_weights.T)
layer_1_values = relu(layer_1_values)
print("layer_1_values")
print(layer_1_values)
layer_2_values = np.dot(layer_1_values, layer_2_weights.T)
print("layer_2_values")
print(layer_2_values)


layer_2_delta = layer_2_values - expected_output
print("layer_2_delta")
print(layer_2_delta)
layer_1_delta = np.dot(layer_2_delta, layer_2_weights)
layer_1_delta = layer_1_delta * relu_deriv(layer_1_values)
print("layer_1_delta")
print(layer_1_delta)


layer_2_weight_delta = np.dot(layer_2_delta.T, layer_1_values)
print("layer_2_weight_delta")
print(layer_2_weight_delta)
layer_1_weight_delta = np.dot(layer_1_delta.T, input)
print("layer_1_weight_delta")
print(layer_1_weight_delta)


layer_2_weights = layer_2_weights - np.dot(alpha, layer_2_weight_delta)
print("layer_2_weights")
print(layer_2_weights)
layer_1_weights = layer_1_weights - np.dot(alpha, layer_1_weight_delta)
print("layer_1_weights")
print(layer_1_weights)



# for iteration in range(10000):
#     error = 0
#
#     for i in range(len(input)):
#         layer_1_values = relu(np.dot(input[i], layer_1_weights.T))
#         layer_2_values = np.dot(layer_1_values, layer_2_weights.T)
#
#         layer_2_delta = layer_2_values - expected_output[i]
#         layer_1_delta = np.dot(layer_2_delta, layer_2_weights) * relu2deriv(layer_1_values)
#
#         layer_2_weight_delta = np.outer(layer_2_delta, layer_1_values)
#         layer_1_weight_delta = np.outer(layer_1_delta, input[i])
#
#         layer_2_weights = layer_2_weights - np.dot(alpha, layer_2_weight_delta)
#         layer_1_weights = layer_1_weights - np.dot(alpha, layer_1_weight_delta)
#
#         error = error + (layer_2_values - expected_output[i]) ** 2
#
#     print("error " + str(sum(error)))
