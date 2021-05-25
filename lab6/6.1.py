# from BitArray2D import BitArray2D
# from random import randint
# from random import getrandbits
# from random import choice
# from bitarray import bitarray
#
#
# number_of_gene = 10
# number_of_chromosome = 10
#
# population = BitArray2D( rows=number_of_gene, columns=number_of_chromosome )
# print(population)
#
# for row in range(number_of_gene):
#     for col in range(number_of_chromosome):
#         los = randint(0, 1)
#         if los == 1:
#             population[row][col] = list('{0:0b}'.format(8))[0]
#
# print(population)
# print(randint(0, 1))

import numpy as np
from random import randint


def fun(arr):
    sum = 0
    for i in arr:
        sum = sum + i
    return sum


number_of_gene = 10
number_of_chromosome = 10

population = np.random.randint(2, size=(number_of_gene, number_of_chromosome))
population = population.tolist()
print(population)

print("selection")

fitness = []
for i in population:
    fitness.append(fun(i))

iter = 0

while max(fitness) != 10:
    print()
    print()
    iter +=1
    print("iteration: " + str(iter))

    arr_fitness = []

    for i in population:
        arr_fitness.append(fun(i))

    print(arr_fitness)
    population_new = []

    for i in range(2):
        index_max = arr_fitness.index(max(arr_fitness))
        population_new.append(population.pop(index_max))
        arr_fitness.pop(index_max)

    print(population)
    print(population_new)

    print("krzyzowanie")

    los = randint(1, 8)
    child = population_new[0][:los]
    child += population_new[1][los:]

    child2 = population_new[1][:los]
    child2 += population_new[0][los:]

    population_new.append(child)
    population_new.append(child2)

    for i in range(6):
        index_max = arr_fitness.index(max(arr_fitness))
        population_new.append(population.pop(index_max))
        arr_fitness.pop(index_max)

    print(population_new)

    print("Mutacja")
    los = randint(1, 100)
    if los <= 60:
        print("yes")
        los_index = randint(1, 9)
        population_new[0][los_index] = 0 if population_new[0][los_index] == 1 else 1
        los_index = randint(1, 9)
        population_new[1][los_index] = 0 if population_new[1][los_index] == 1 else 1

    print(population_new)

    population = population_new

    fitness = []
    for i in population:
        fitness.append(fun(i))
    print("fitness " + str(fitness))
