import copy
import numpy as np
from random import randint
import math
from math import inf
from math import *


def fun_sum(arr):
    sum = 0
    for i in arr:
        sum = sum + i
    return sum


def fun_fitness(arr):
    lenth = 0
    for i in range(len(arr) - 1):
        lenth += math.sqrt(math.pow(X[arr[i]] - X[arr[i + 1]], 2) + math.pow(Y[arr[i]] - Y[arr[i + 1]], 2))

    lenth += math.sqrt(math.pow(X[arr[len(arr) - 1]] - X[arr[0]], 2) + math.pow(Y[arr[len(arr) - 1]] - Y[arr[0]], 2))
    return 10000 - lenth


def get_chromosome():
    l = []
    for i in range(25):
        l.append(i)
    l2 = []

    while len(l) > 0:
        los = randint(0, len(l) - 1)
        l2.append(l.pop(los))

    return l2


X = [119, 37, 197, 85, 12, 100, 81, 121, 85, 80, 91, 106, 123, 40, 78, 190, 187, 37, 17, 67, 78, 87, 184, 111, 66]
Y = [38, 38, 55, 165, 50, 53, 142, 237, 145, 197, 176, 55, 57, 81, 125, 46, 40, 107, 11, 56, 133, 23, 197, 12, 178]

number_of_gene = 25
number_of_chromosome = 100

population = []
for g in range(number_of_chromosome):
    population.append(get_chromosome())
print(population)

for f in range(1000):
    print()
    print()
    print("iteration: " + str(f + 1))

    arr_fitness = []

    for i in population:
        arr_fitness.append(fun_fitness(i))

    print(arr_fitness)

    population_new = []

    # elitarnosc
    temp = copy.deepcopy(population)
    temp_fit = copy.deepcopy(arr_fitness)
    for m in range(20):
        index = temp_fit.index(max(temp_fit))
        population_new.append(temp.pop(index))
        temp_fit.pop(index)

    print(population_new)


    # -------------------------------------------------------------------------------------------

    print("Selekcja")

    population_best = []

    for i in range(80):
        max_value = fun_sum(arr_fitness)
        los = randint(1, int(max_value))
        index = 0
        while los > 0:
            los -= arr_fitness[index]
            index += 1

        index -= 1
        population_best.append(population[index])

    print(population)
    print(population_best)


    # ------------------------------------------------------------------------------------------

    print("Krzyzowanie")

    for l in range(80-1):
        index_start = randint(1, int(number_of_gene/2))
        long = randint(1, int(number_of_gene/3))
        child = [np.random.randint(-1, 0) for i in range(number_of_gene)]
        child[index_start:index_start+long] = population_best[l][index_start:index_start+long]
        temp = []
        for p in population_best[l+1]:
            if p not in child:
                temp.append(p)
        for b in range(number_of_gene):
            if child[b]==-1:
                child[b] = temp.pop(0)
        population_new.append(child)

    population_new.append(population_best[len(population_best) - 1])

    population = population_new
    print(population)


    # ---------------------------------------------------------------------------------------------

    print("Mutacja")

    for k in range(20, number_of_chromosome):
        for l in range(number_of_gene-1):
            los = randint(1, 100)
            if los <= 1:
                temp = population[k][l]
                population[k][l] = population[k][l+1]
                population[k][l + 1] = temp


    print(population)




fitness = []
for i in population:
    fitness.append(fun_fitness(i))

print()
wynik = population[fitness.index(max(fitness))]
print(wynik)
print("lenth = " + str(10000 - fun_fitness(wynik)))


# [16, 23, 0, 12, 11, 5, 21, 19, 1, 18, 4, 13, 17, 14, 20, 6, 8, 3, 10, 24, 9, 7, 22, 2, 15]
# lenth = 839.2736739990905
# [8, 14, 17, 1, 19, 5, 11, 12, 16, 15, 2, 0, 23, 21, 18, 4, 13, 20, 6, 10, 22, 7, 9, 24, 3]
# lenth = 999.6097337905649





