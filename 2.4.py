import numpy as np


input_values = np.array([0.91, 0.82, 0.05])

weight_matrix = np.array([[0.1, 0.1, 0.3],
                          [0.1, 0.2, 0.0],
                          [0.1, 0.2, 0.0],
                          [0.0, 0.3, 0.1]])

expected_output = np.array([0, 0, 0, 0])
expected_output[4-1] = 1
alpha = 0.01


infile = open("data_train.txt", 'r')

train_data = np.array([line.split() for line in infile])
ID = train_data[:, 3]
# print(ID)
train_data = train_data[:, 0:3]
# print(train_data)

input_values = train_data.astype(np.float64)
ID = ID.astype(np.int_)

for i in range(10):
    for ind in range(len(ID)):
        # print(i+1)
        expected_output = np.array([0, 0, 0, 0])
        expected_output[ID[ind] - 1] = 1
        # print("expected_output "+ str(expected_output))
        output_value = np.dot(input_values[ind], weight_matrix.T)
        weight_matrix = weight_matrix - np.dot(alpha, np.outer(output_value - expected_output, input_values[ind]))

        # print("weight " + str(weight_matrix))
        # print("output " + str(output_value))
        # print("output " + str(np.rint(output_value)))
        # error = (output_value - expected_output) ** 2
        # print("error " + str(error))
        # print("sum " + str(np.sum(error)))
        # print()




infile2 = open("data_test.txt", 'r')

test_data = np.array([line.split() for line in infile2])
print(test_data)

ID2 = test_data[:, 3]
# print(ID)
test_data = test_data[:, 0:3]
# print(train_data)

input_values = test_data.astype(np.float64)
ID2 = ID2.astype(np.int_)

prawda = 0
nieprawda = 0

for ind in range(len(ID2)):
    expected_output = np.array([0, 0, 0, 0])
    expected_output[ID2[ind] - 1] = 1
    print("expected_output " + str(expected_output))
    output_value = np.dot(input_values[ind], weight_matrix.T)
    print("output " + str(output_value))
    # output_value = output_value.astype(np.int_)
    # print("output " + str(np.rint(output_value)))
    # print(expected_output==output_value)
    # print(ID2[ind])
    # print(np.argmax(output_value) + 1)
    if ID2[ind] == (np.argmax(output_value) + 1):
        prawda +=1
    else:
        nieprawda +=1
    print()



print("poprawne zgadniete siecia: " + str(prawda))
print("niezgadniete: " + str(nieprawda))





