import numpy as np
from numpy import exp, array, random, dot, tanh
import pickle
import sys
import numpy as np




input_neurons = 784
tab_hidden_neurons = 40
output_neurons = 10



alpha = 0.01
number_images_train = 10000
number_images_test = 10000
iterations = 50

batch_size = 1



pixels_per_image, num_labels = (784, 10)

input_rows = 28
input_cols = 28
kernel_rows = 3
kernel_cols = 3
num_kernels = 16


# network = NeuralNetwork(input_neurons, tab_hidden_neurons, output_neurons, alpha)

file_labels_train = open("../train-labels.idx1-ubyte", "rb")
print(int.from_bytes(file_labels_train.read(4), "big"))
print(int.from_bytes(file_labels_train.read(4), "big"))
print()

file_images_train = open("../train-images.idx3-ubyte", "rb")
print(int.from_bytes(file_images_train.read(4), "big"))
print(int.from_bytes(file_images_train.read(4), "big"))
print(int.from_bytes(file_images_train.read(4), "big"))
print(int.from_bytes(file_images_train.read(4), "big"))

all_images = []
all_label = []

for i in range(number_images_train):

    images = []
    for k in range(28 * 28):
        images.append(int.from_bytes(file_images_train.read(1), "big"))

    images = np.array(images)
    images = images / 255

    label = int.from_bytes(file_labels_train.read(1), "big")
    exp = np.zeros(10)
    exp[label] = 1

    all_images.append(images)
    all_label.append(exp)



print()
# print("--------------------------test---------------------------")

file_labels_test = open("../t10k-labels.idx1-ubyte", "rb")
print(int.from_bytes(file_labels_test.read(4), "big"))
print(int.from_bytes(file_labels_test.read(4), "big"))
print()

file_images_test = open("../t10k-images.idx3-ubyte", "rb")
print(int.from_bytes(file_images_test.read(4), "big"))
print(int.from_bytes(file_images_test.read(4), "big"))
print(int.from_bytes(file_images_test.read(4), "big"))
print(int.from_bytes(file_images_test.read(4), "big"))
print()

all_images_test = []
all_label_test = []

for i in range(number_images_test):
    images = []
    for k in range(28 * 28):
        images.append(int.from_bytes(file_images_test.read(1), "big"))

    images = np.array(images)
    images = images / 255

    label = int.from_bytes(file_labels_test.read(1), "big")
    exp = np.zeros(10)
    exp[label] = 1

    all_images_test.append(images)
    all_label_test.append(exp)

all_images = np.array(all_images)
all_images = all_images.reshape(number_images_train, 28 * 28)

print("all_images")
print(all_images.shape)
print(all_images)
print()

all_label = np.array(all_label)

print("all_label")
print(all_label.shape)
print(all_label)
print()

all_images_test = np.array(all_images_test)
all_images_test = all_images_test.reshape(number_images_test, 28 * 28)

print("all_images_test")
print(all_images_test.shape)
print(all_images_test)
print()

all_label_test = np.array(all_label_test)

print("all_label_test")
print(all_label_test.shape)
print(all_label_test)



def relu( x):
    return (x > 0) * x


def relu_deriv(output):
    return output > 0


def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)




hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels

kernels = 0.02 * np.random.random((kernel_rows * kernel_cols, num_kernels)) - 0.01

layer_2_weights = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1



def pooling(input_image, F=2, S=2):

    pool_out_image = np.zeros((np.uint16((input_image.shape[0]-F)/S + 1), np.uint16((input_image.shape[1]-F)/S)+1))

    row = 0
    for x in np.arange(0, input_image.shape[0]-F-1, S):
        col = 0
        for y in np.arange(0, input_image.shape[1]-F-1, S):
            pool_out_image[row, col] = np.max([input_image[x:x+F, y:y+F]])
            col += 1
        row += 1

    return pool_out_image




def get_image_section(layer, row_from, row_to, col_from, col_to):
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to - row_from, col_to - col_from)


for j in range(iterations):
    correct_cnt = 0

    # ----------------------------------- TRAIN ----------------------------------------------------
    for i in range(int(len(all_images) / batch_size)):

        batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
        layer_0 = all_images[batch_start:batch_end]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)

        all_sections_of_image = list()

        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                section = get_image_section(layer_0, row_start, row_start + kernel_rows, col_start, col_start + kernel_cols)
                all_sections_of_image.append(section)

        image_sections = np.concatenate(all_sections_of_image, axis=1)
        in_shape = image_sections.shape
        flattened_image_sections = image_sections.reshape(in_shape[0] * in_shape[1], -1)

        kernel_output = flattened_image_sections.dot(kernels)




        layer_1_values = relu(kernel_output.reshape(in_shape[0], -1))


        all_sections_of_image_2 = list()

        for point in range(layer_1_values.shape[0]):
            # temp2 = pooling(layer_1_values[point].reshape(100, -1), 2, 2)
            temp_reshape = layer_1_values[point].reshape(100, -1)
            all_sections_of_image_2.append(pooling(temp_reshape, 2, 2))

        pooling_layout = np.array(all_sections_of_image_2)
        pooling_layout = pooling_layout.reshape(pooling_layout.shape[0], -1)


        dropout_mask = np.random.randint(2, size=layer_1_values.shape)
        layer_1_values = layer_1_values * dropout_mask * 2


        temp = np.dot(layer_1_values, layer_2_weights)
        layer_2_values = softmax(temp)



        for k in range(batch_size):
            correct_cnt += int(np.argmax(layer_2_values[k:k + 1]) == np.argmax(all_label[batch_start + k:batch_start + k + 1]))


        layer_2_delta = (all_label[batch_start:batch_end] - layer_2_values) / (batch_size * layer_2_values.shape[0])

        layer_1_delta = layer_2_delta.dot(layer_2_weights.T) * relu_deriv(layer_1_values)
        layer_1_delta *= dropout_mask


        layer_2_weights = layer_2_weights + alpha * layer_1_values.T.dot(layer_2_delta)

        layer_1_delta_reshape = layer_1_delta.reshape(kernel_output.shape)
        kernels += alpha * (flattened_image_sections.T.dot(layer_1_delta_reshape))


    # ----------------------------------- TEST ----------------------------------------------------
    test_correct_cnt = 0

    for i in range(len(all_images_test)):

        layer_0 = all_images_test[i:i + 1]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)


        all_sections_of_image = list()
        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                section = get_image_section(layer_0, row_start, row_start + kernel_rows, col_start, col_start + kernel_cols)
                all_sections_of_image.append(section)

        image_sections = np.concatenate(all_sections_of_image, axis=1)
        in_shape = image_sections.shape
        flattened_image_sections = image_sections.reshape(in_shape[0] * in_shape[1], -1)

        kernel_output = flattened_image_sections.dot(kernels)
        layer_1_values = relu(kernel_output.reshape(in_shape[0], -1))

        layer_2_values = np.dot(layer_1_values, layer_2_weights)

        test_correct_cnt += int(np.argmax(layer_2_values) == np.argmax(all_label_test[i:i + 1]))




    # ----------------------------------- RESULT ----------------------------------------------------
    if j % 1 == 0:
        sys.stdout.write(
            "\n" + "I:" + str(j) + " Test-Acc:" + str(test_correct_cnt / float(len(all_images_test))) + " Train-Acc:" + str(
                correct_cnt / float(len(all_images))))
