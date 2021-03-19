from numpy import exp, array, random, dot, tanh
import numpy as np


input_values = np.array([8.5, 0.65, 1.2])
print(input_values)

weight_matrix = 2 * random.random((3, 1)) - 1
print(weight_matrix)


output_values = np.dot(input_values, weight_matrix)
print(output_values)

