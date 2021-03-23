import numpy as np


input_values = np.array([[8.5, 0.65, 1.2],
                         [9.5, 0.8, 1.3],
                         [9.9, 0.8, 0.5],
                         [9.0, 0.9, 1.0]])


weight_matrix = np.array([[0.1, 0.1, -0.3],
                          [0.1, 0.2, 0.0],
                          [0.0, 1.3, 0.1]])

expected_output = np.array([[0.1, 1, 0.1],
                           [0, 1, 0],
                           [0, 0, 0.1],
                           [0.1, 1, 0.2]])

alpha = 0.01




for i in range(50):
    er = 0
    for ind in range(len(input_values)):
        print(i+1)
        output_value = np.dot(input_values[ind], weight_matrix.T)
        weight_matrix = weight_matrix - np.dot(alpha, np.outer(output_value - expected_output[ind], input_values[ind]))

        print("weight " + str(weight_matrix))
        print("output " + str(output_value))
        error = (output_value - expected_output[ind]) ** 2
        print("error " + str(error))
        # print("sum " + str(np.sum(error)))
        er = er + error

    print("error " + str(er))
    print("error " + str(sum(er)))