
import numpy as np
from random import randint

def fun_sum(arr):
    sum = 0
    for i in arr:
        sum = sum + i
    return sum

def fun_fitness(arr):
    sum = 0
    ret = 0
    for i in range(len(arr)):
        sum += arr[i]*wagi[i]
        ret +=arr[i]*wartosci[i]
    if sum >35:
        return 0
    return ret


wagi = [3, 13, 10, 9, 7, 1, 8, 8, 2, 9]
wartosci = [266, 442, 671, 526, 388, 245, 210, 145, 126, 322]

number_of_gene = 10
number_of_chromosome = 8

population = np.random.randint(2, size=(number_of_chromosome, number_of_gene))
population = population.tolist()
print(population)




for f in range(80):
    print()
    print()
    print("iteration: " + str(f+1))

    arr_fitness = []

    for i in population:
        arr_fitness.append(fun_fitness(i))

    print(arr_fitness)

    population_new = []
    population_new.append(population[arr_fitness.index(max(arr_fitness))])

    # ---------------------------------------------------------------------------------
    print("Selekcja")


    population_best = []


    for i in range(number_of_chromosome-1):
        max_value = fun_sum(arr_fitness)
        los = randint(1, max_value)
        index = 0
        while los>0:
            los -= arr_fitness[index]
            index += 1

        index -= 1
        population_best.append(population[index])


    print(population)
    print(population_best)

    # ---------------------------------------------------------------------------------
    print("Krzyzowanie")


    for l in range(len(population_best)-1):
        los = randint(1, number_of_gene-1)
        child = population_best[l][:los]
        child += population_best[l+1][los:]
        population_new.append(child)

    population_new.append(population_best[len(population_best)-1])


    population = population_new
    print(population)
    # ---------------------------------------------------------------------------------
    print("Mutacja")

    for k in range(1, number_of_chromosome):
        for l in range(number_of_gene):
            los = randint(1, 100)
            if los <= 5:
                population[k][l] = 0 if population[k][l] == 1 else 1

    print(population)


fitness = []
for i in population:
    fitness.append(fun_fitness(i))

print()
wynik = population[fitness.index(max(fitness))]
print(wynik)
print("waga = " + str(fun_fitness(wynik)))

for el in range(10):
    if wynik[el]!=0:
        print(el)

