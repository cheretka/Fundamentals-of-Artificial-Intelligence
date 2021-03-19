import numpy as np


input_values = np.array([[8.5, 0.65, 1.2],
                         [9.5, 0.8, 1.3],
                         [9.9, 0.8, 0.5],
                         [9.0, 0.9, 1.0]])
print(input_values)

weight_matrix = np.array([[0.1, 0.1, -0.3],
                          [0.1, 0.2, 0.0],
                          [0.0, 1.3, 0.1]]).T
print(weight_matrix)

output_values = np.dot(input_values, weight_matrix)
print(output_values)

