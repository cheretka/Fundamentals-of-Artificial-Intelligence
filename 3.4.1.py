import numpy as np
from numpy import exp, array, random, dot, tanh
import pickle


def predict( input, layer_1_weights, layer_2_weights):
    layer_1_values = sigmoid(np.dot(input, layer_1_weights.T))
    layer_2_values = np.dot(layer_1_values, layer_2_weights.T)
    return layer_2_values

def sigmoid(input):
    return 1 / (1 + np.exp(-input))



if __name__ == "__main__":

    layer_1_weights = np.loadtxt("weights1.txt").reshape(300, 74073)

    print(layer_1_weights)

    layer_2_weights = np.loadtxt("weights2.txt").reshape(1, 300)

    print(layer_2_weights)

    unique_list = np.loadtxt("unique_list.txt").reshape(1, 300)


    all_rewiews = []
    numberOfAllRewiews = 0
    with open("rewiews.txt") as fp:
        Lines = fp.readlines()
        for line in Lines:
            all_rewiews.append(line)
            numberOfAllRewiews +=1
    print(numberOfAllRewiews)

    numberOfRewiews_train = int(numberOfAllRewiews*24 /25)
    numberOfRewiews_test = int(numberOfAllRewiews /25)
    print(numberOfRewiews_train)
    print(numberOfRewiews_test)

    rewiews_train = all_rewiews[ 0:numberOfRewiews_train]
    rewiews_test = all_rewiews[numberOfRewiews_train : numberOfRewiews_train+numberOfRewiews_test]

    all_labels = []
    numberOfAllLabels = 0
    with open("labels.txt") as fp:
        Lines = fp.readlines()
        for line in Lines:
            if line == "negative\n":
                all_labels.append(0)
            else:
                all_labels.append(1)
            numberOfAllLabels +=1
    print(numberOfAllLabels)

    labels_train = all_labels[0:numberOfRewiews_train]
    labels_test = all_labels[numberOfRewiews_train: numberOfRewiews_train + numberOfRewiews_test]
    print(len(labels_train))
    print(len(labels_test))


##################################################################

    print("\n\n\n")
    unique_list = []
    licz = 0
    for rewiew in all_rewiews:
        all_words = rewiew.split()
        for word in all_words:
            if word not in unique_list:
                unique_list.append(word)
        licz += 1
        if licz % 100 == 0:
            print(licz)

    # print(unique_list)
    print("len(unique_list)=")
    print(len(unique_list))

    lenInput = len(unique_list)


######################## train ############################

    print("\n\n\n")

    good_points = 0
    bad_points = 0
    licz = 0
    for k in range(numberOfRewiews_test):
        wektor = []
        # print(rewiews_test[k])

        for i in range(lenInput):
            if unique_list[i] in rewiews_test[k]:
                wektor.append(1)
            else:
                wektor.append(0)

        # print(wektor)
        output2 = predict(wektor, layer_1_weights, layer_2_weights)
        output = round(output2[0])
        if output == labels_test[k]:
            good_points += 1
        else:
            bad_points += 1

        licz += 1
        if licz % 100 == 0:
            print(licz)

    print("good_points")
    print(good_points)
    print("bad_points")
    print(bad_points)