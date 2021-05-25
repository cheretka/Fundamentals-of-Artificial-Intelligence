import numpy as np

def relu(x):
    return (x > 0) * x


input = np.array([[8.5, 0.65, 1.2],
                 [9.5, 0.8, 1.3],
                 [9.9, 0.8, 0.5],
                 [9.0, 0.9, 1.0]])

layer_1_weights = np.array([[0.1, 0.2, -0.1],
                            [-0.1, 0.1, 0.9],
                            [0.1, 0.4, 0.1]]).T

layer_2_weights = np.array([[0.3, 1.1, -0.3],
                            [0.1, 0.2, 0.0],
                            [0.0, 1.3, 0.1]]).T


for i in range(len(input)):
    output = relu(np.dot(input[i], layer_1_weights))
    output = np.dot(output, layer_2_weights)
    print(output)







