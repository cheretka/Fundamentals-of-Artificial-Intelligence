import numpy as np
from numpy import exp, array, random, dot, tanh
import pickle
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

input_image = np.array([[1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1],
                        [0, 0, 1, 1, 0],
                        [0, 1, 1, 0, 0]])

filter = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])

output_image = np.array([[4, 3, 4],
                         [2, 4, 3],
                         [2, 3, 4]])
S = 1
pad = 1


def fun_splotu_obrazu(input_image, filter, stride, zero_padding):
    input_image_pad = np.pad(input_image, (zero_padding,), 'constant', constant_values=(0,))
    print("input_image_pad")
    print(input_image_pad)
    row_output = int((input_image_pad.shape[0] - filter.shape[0]) / stride + 1)
    col_output = int((input_image_pad.shape[1] - filter.shape[1]) / stride + 1)

    sects = list()
    for row_start in range(row_output):
        for col_start in range(col_output):
            row_from = row_start * stride
            row_to = row_start * stride + filter.shape[0]
            col_from = col_start * stride
            col_to = col_start * stride + filter.shape[1]

            section = input_image_pad[row_from:row_to, col_from:col_to]
            section = section.T.reshape(1, -1)

            sects.append(section)

    image_sections = np.concatenate(sects, axis=0)

    kernel_layer = np.dot(image_sections, filter.reshape(1, -1).T)
    image_output = kernel_layer.reshape(row_output, col_output)

    return image_output


result = fun_splotu_obrazu(input_image, filter, S, pad)
print("expected")
print(output_image)
print("got")
print(result)
