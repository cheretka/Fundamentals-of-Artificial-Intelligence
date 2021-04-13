import numpy as np


def relu(x):
    return (x > 0) * x

def relu2deriv(output):
    return output>0



layer_1_weights = np.array([[0.1, 0.1, 0.3],
                           [0.1, 0.2, 0.0],
                           [0.1, 0.2, 0.0]])

layer_2_weights = np.array([[0.1, 0.1, 0.3],
                           [0.1, 0.2, 0.0],
                           [0.1, 0.2, 0.0],
                           [0.0, 0.3, 0.1]])


expected_output = np.array([0, 0, 0, 0])
expected_output[4-1] = 1
alpha = 0.01


infile = open("data_train.txt", 'r')
train_data = np.array([line.split() for line in infile])
ID = train_data[:, 3]
train_data = train_data[:, 0:3]


input_values = train_data.astype(np.float64)
ID = ID.astype(np.int_)


for i in range(15):
    for ind in range(len(ID)):
        expected_output = np.array([0, 0, 0, 0])
        expected_output[ID[ind] - 1] = 1

        layer_1_values = relu(np.dot(input_values[ind], layer_1_weights.T))
        layer_2_values = np.dot(layer_1_values, layer_2_weights.T)

        layer_2_delta = layer_2_values - expected_output
        layer_1_delta = np.dot(layer_2_delta, layer_2_weights) * relu2deriv(layer_1_values)

        layer_2_weight_delta = np.outer(layer_2_delta, layer_1_values)
        layer_1_weight_delta = np.outer(layer_1_delta, input_values[ind])

        layer_2_weights = layer_2_weights - np.dot(alpha, layer_2_weight_delta)
        layer_1_weights = layer_1_weights - np.dot(alpha, layer_1_weight_delta)







infile2 = open("data_test.txt", 'r')

test_data = np.array([line.split() for line in infile2])
print(test_data)

ID2 = test_data[:, 3]
test_data = test_data[:, 0:3]

input_values = test_data.astype(np.float64)
ID2 = ID2.astype(np.int_)

prawda = 0
nieprawda = 0


for ind in range(len(ID2)):
    expected_output = np.array([0, 0, 0, 0])
    expected_output[ID2[ind] - 1] = 1

    layer_1_values = relu(np.dot(input_values[ind], layer_1_weights.T))
    output_value = np.dot(layer_1_values, layer_2_weights.T)

    output = np.array([0, 0, 0, 0])
    output[np.argmax(output_value)] = 1

    if (expected_output == output).all():
        prawda += 1
    else:
        nieprawda += 1



print("+  " + str(prawda))
print("-  " + str(nieprawda))





