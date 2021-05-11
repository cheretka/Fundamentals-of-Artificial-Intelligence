import numpy as np
from numpy import exp, array, random, dot, tanh
import pickle
import sys

input_image = np.array([[8.5, 0.65, 1.2],
                        [9.5, 0.8, 1.3],
                        [9.9, 0.8, 0.5],
                        [9.0, 0.9, 1.0]])

expected_output = np.array([0, 1])

alpha = 0.01

kwadrat = 3

kernel_1_weights = np.array([0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1])
kernel_2_weights = np.array([0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1])

kernels = np.concatenate((kernel_1_weights.reshape(1, 9), kernel_2_weights.reshape(1, 9)))

layer_1_weights = kernels
layer_2_weights = np.array([[0.1, -0.2, 0.1, 0.3],
                            [0.2, 0.1, 0.5, -0.3]])




def get_image_section(layer, row_from, row_to, col_from, col_to):
    section = layer[row_from:row_to, col_from:col_to]
    return section.T.reshape(1, -1)





sects = list()
for row_start in range(2):
    for col_start in range(1):
        sect = get_image_section(input_image, row_start, row_start + kwadrat, col_start, col_start + kwadrat)
        sects.append(sect)

image_sections = np.concatenate(sects, axis=0)

kernel_layer = np.dot(image_sections, kernels.T)
kernel_layer_flatten = np.concatenate(kernel_layer, axis=None)

layer_2_values = np.dot(kernel_layer_flatten, layer_2_weights.T)

layer_2_delta = layer_2_values - expected_output
layer_1_delta = np.dot(layer_2_delta, layer_2_weights)
layer_1_delta_reshaped = np.reshape(layer_1_delta, (-1, 2))

layer_2_weights_delta = np.reshape(layer_2_delta, (-1, 1)) * kernel_layer_flatten
layer_1_weights_delta = np.dot(layer_1_delta_reshaped.T, image_sections)

layer_2_weights = layer_2_weights - np.dot(alpha, layer_2_weights_delta)
layer_1_weights = layer_1_weights - np.dot(alpha, layer_1_weights_delta)
