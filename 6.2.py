
import numpy as np
from random import randint

def fun_sum(arr):
    sum = 0
    for i in arr:
        sum = sum + i
    return sum

def fun_fitness(arr):
    a = arr[0]*8 + arr[1]*4 + arr[2]*2 + arr[3]*1
    b = arr[4]*8 + arr[5]*4 + arr[6]*2 + arr[7]*1
    sum = 2*(a**2) + b
    return 500 - abs(33-sum)


number_of_gene = 8
number_of_chromosome = 10

population = np.random.randint(2, size=(number_of_chromosome, number_of_gene))
population = population.tolist()
print(population)


fitness = []
for i in population:
    fitness.append(fun_fitness(i))

iter = 0

while max(fitness) != 500:
    print(max(fitness))
    print()
    print()
    iter += 1
    print("iteration: " + str(iter))

    arr_fitness = []

    for i in population:
        arr_fitness.append(fun_fitness(i))

    print(arr_fitness)

    # ---------------------------------------------------------------------------------
    print("Selekcja")


    population_best = []


    for i in range(5):
        max_value = fun_sum(arr_fitness)
        los = randint(1, max_value)
        index = 0
        while los>0:
            los -= arr_fitness[index]
            index += 1

        index -= 1
        population_best.append(population.pop(index))
        arr_fitness.pop(index)

    print(population)
    print(population_best)

    # ---------------------------------------------------------------------------------
    print("Krzyzowanie")

    population_new = []
    for l in range(4):
        los = randint(1, number_of_gene-1)
        child = population_best[l][:los]
        child += population_best[l+1][los:]
        population_new.append(child)

    population_new.append(population_best[4])


    for smth in population_new:
        population.append(smth)


    print(population)


    # ---------------------------------------------------------------------------------
    print("Mutacja")

    for k in range(10):
        los = randint(1, 100)
        if los <= 10:
            los_index = randint(0, number_of_gene-1)
            population[k][los_index] = 0 if population[k][los_index] == 1 else 1

    print(population)


    # ---------------------------------------------------------------------------------
    fitness = []
    for i in population:
        fitness.append(fun_fitness(i))
    print("fitness " + str(fitness))




print()
wynik = population[fitness.index(max(fitness))]
print(wynik)
a = wynik[0]*8 + wynik[1]*4 + wynik[2]*2 + wynik[3]*1
b = wynik[4]*8 + wynik[5]*4 + wynik[6]*2 + wynik[7]*1
sum = 2*(a**2) + b
print("a = " + str(a))
print("b = " + str(b))


# 3 15 \ 4 1