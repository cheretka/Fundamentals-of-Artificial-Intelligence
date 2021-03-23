from numpy import exp, array, random, dot, tanh
import numpy as np


input_values = np.array([[8.5, 0.65, 1.2],
                         [9.5, 0.8, 1.3],
                         [9.9, 0.8, 0.5],
                         [9.0, 0.9, 1.0]])

weight_matrix = np.array([0.1, 0.2, -0.1])

expected_output = np.array([1., 1., 0., 1.]).T

alpha = 0.01



# for i in range(1000):
#     print(i+1)
#     output_value = np.dot(input_values, weight_matrix.T)
#     delta = output_value - expected_output
#     weight_delta = np.dot(delta, input_values)
#     weight_matrix = weight_matrix - np.dot(alpha, weight_delta)
#
#     # print("delta " + str(delta))
#     # print("weight_delta " + str(weight_delta))
#     # print("weight " + str(weight_matrix))
#     print("output " + str(output_value))
#     error = (output_value - expected_output) ** 2
#     print("error " + str(error))
#     print()

er = 0

for i in range(1000):
    print(i+1)
    output_value = np.dot(input_values[0], weight_matrix.T)
    delta = output_value - expected_output[0]
    weight_delta = np.dot(delta, input_values[0])
    weight_matrix = weight_matrix - np.dot(alpha, weight_delta)

    # print("delta " + str(delta))
    # print("weight_delta " + str(weight_delta))
    # print("weight " + str(weight_matrix))
    print("output " + str(output_value))
    error = (output_value - expected_output[0]) ** 2
    print("error " + str(error))
    er = er + error
    print()

print(er)


# for i in range(1000):
#     for ind in range(4):
#         print(i+1)
#         output_value = np.dot(input_values[ind], weight_matrix)
#         delta = output_value - expected_output[ind]
#         weight_delta = np.dot(delta, input_values[ind])
#         weight_matrix = weight_matrix - np.dot(alpha, weight_delta)
#
#         print("delta " + str(delta))
#         print("weight_delta " + str(weight_delta))
#         print("weight " + str(weight_matrix))
#         print("output " + str(output_value))
#         error = (output_value - expected_output[ind]) ** 2
#         print("error " + str(error))
#         print()