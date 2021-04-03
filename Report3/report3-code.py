import numpy as np
import random
import time
import matplotlib.pyplot as plt

#----- is List correct or not -------------
def R_column(A, y):
    if np.sum(A.T[y]):
        return True
    return False

def R_row(A,x):
    if np.sum(A[x]):
        return True
    return False


def R_diag(A, x, y):
    n = A.shape[0]
    result = False
    i = x
    j = y
    while i < n and j < n:
        if A[i][j] == 1:
            result = True
            break
        i += 1
        j += 1
    if result:
        return True
    i = x - 1
    j = y - 1
    while i > -1 and j > -1 :
        if A[i][j] == 1:
            result = True
            break
        i -= 1
        j -= 1
    if result:
        return True
    i = x + 1
    j = y - 1
    while  i < n and j > -1 :
        if A[i][j] == 1:
            result = True
            break
        i += 1
        j -= 1
    if result:
        return True
    i = x - 1
    j = y + 1
    while  i > -1 and j < n :
        if A[i][j] == 1:
            result = True
            break
        i -= 1
        j += 1
    return result


def R_collision(A, x, y):
    return R_column(A, y) or R_row(A, x) or R_diag(A, x, y)

def IsAnswer(L):
    n = len(L)
    A = np.zeros((n, n))
    for i in range(n):
        A[L[i]][i] = 1
    if np.sum(A) != n:
        return False
    for i in range(n):
        for j in range(n):
            if A[i][j] == 1:
                A[i][j] = 0
                if R_collision(A, i, j):
                    A[i][j] = 1
                    return False
                A[i][j] = 1
    return True


# --------- counting collisions -------
def row(A,x, y):
    return np.sum(A[x]) - A[x][y]


def diag(A, x, y):
    numTarget = 0
    n = len(A)
    i = x + 1
    j = y + 1
    while i < n and j < n:
        if A[i][j] == 1:
            numTarget += 1
        i += 1
        j += 1
    i = x - 1
    j = y - 1
    while i > -1 and j > -1 :
        if A[i][j] == 1:
            numTarget += 1
        i -= 1
        j -= 1
    i = x + 1
    j = y - 1
    while  i < n and j > -1 :
        if A[i][j] == 1:
            numTarget += 1
        i += 1
        j -= 1
    i = x - 1
    j = y + 1
    while  i > -1 and j < n :
        if A[i][j] == 1:
            numTarget += 1
        i -= 1
        j += 1
    return numTarget


def collision(L, x, y):
    n = len(L)
    A = np.zeros((n, n))
    for i in range(n):
        A[L[i]][i] = 1
    return row(A, x, y) + diag(A, x, y)

def collision_pair(L):
    n = len(L)
    result = 0
    for i in range(n):
        result += collision(L, L[i], i)
    return result / 2

# A = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1],
#              [0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
#
# L = np.array([4, 5, 6, 3, 4, 5, 6, 5])
#
# print(collision_pair(L))
# x = np.flatnonzero(A == np.argmin(A))
# print(x)

#------------ Hill climbing ------------------
def hill_climbing(n, max_iter):
    L = np.random.randint(0,n-1,(n))
    A = np.zeros((n,n))
    for i in range(n):
        A[L[i]][i] = 1
    h = collision_pair(L)
    iterate = 0
    L_result = L
    h_mat = np.ones((n, n))*100
    while iterate < max_iter and h > 0:
        h_mat = np.ones((n, n))*100
        for i in range(n):
            for j in range(n):
                if L_result[j] != i:
                    h_mat[i][j] = h + collision(L_result, i, j) - collision(L_result, L_result[j], j)
        h = np.min(h_mat)
        min_index = np.flatnonzero(h_mat == h)
        change = random.choice(min_index)
        L_result[change%n] = int(change/n)
        iterate += 1
        # print('iter: ', iterate, ', List = ', L_result, ', h = ', h)

    # print('List is correct? ', IsAnswer(L_result))
    # print(h_mat)

    return L_result


#----------- Genetic algorithm ---------------
def Genetic_algorithm(n, num_population, max_iter):
    optimal = n*(n-1)/2
    population = np.random.randint(0,n-1,(num_population, n))
    # print(population)
    iterate = 0
    result = -1
    while iterate < max_iter:
        iterate += 1
        fitness = np.zeros((num_population, 1))
        for i in range(num_population):
            fitness[i] = optimal - collision_pair(population[i])
            if fitness[i] == optimal:
                result = i
        if result != -1:
            break
        fitness_percent = np.cumsum(fitness / np.sum(fitness))
        # print(fitness)
        # print(fitness_percent)
        rand = np.random.random(num_population)
        # print(rand)
        population_temp = np.zeros((num_population, n))
        for i in range(num_population):
            for j in range(num_population):
                if rand[i] < fitness_percent[j]:
                    population_temp[i] = population[j]
                    break
        # print(population_temp)
        for i in range(int(num_population/2)):
            index = random.randint(0, n-2)
            population[2 * i][0:index + 1] = population_temp[2 * i][0:index + 1]
            population[2 * i][index + 1:n] = population_temp[2 * i + 1][index + 1:n]
            population[2 * i + 1][0:index + 1] = population_temp[2 * i + 1][0:index + 1]
            population[2 * i + 1][index + 1:n] = population_temp[2 * i][index + 1:n]
        # print(population)

        # mutation
        for i in range(num_population):
            for j in range(n):
                if random.random() < 0.1:
                    population[i][j] = random.randint(0, n-1)
        # print(population)

    if result != -1:
        return population[result]
    else:
        fitness = np.zeros((num_population, 1))
        for i in range(num_population):
            fitness[i] = optimal - collision_pair(population[i])
            if fitness[i] == optimal:
                result = i
        if result != -1:
            return population[result]
        else:
            x = np.flatnonzero(fitness == max(fitness))
            return population[x[0]]

#---------------------- main -------------------------
time_hill = []
time_genetic = []
correct_hill = []
correct_genetic = []
N = range(4,16)
num_computation = 100
for n in N:
    print('n = ', n, ': processing ...')
    temp_hill = 0
    cnt_hill = 0
    temp_genetic = 0
    cnt_genetic = 0
    for i in range(100):
        tik = time.time()
        L = hill_climbing(n, 20)
        # L = Genetic_algorithm(n, 20, 50)
        tok = time.time()
        temp_hill += (tok - tik) * 1000
        if IsAnswer(L):
            cnt_hill += 1

        tik = time.time()
        L = Genetic_algorithm(n, 10, 20)
        tok = time.time()
        temp_genetic += (tok - tik) * 1000
        if IsAnswer(L):
            cnt_genetic += 1


    time_hill.append(temp_hill/num_computation)
    correct_hill.append(cnt_hill * 100 / num_computation)
    time_genetic.append(temp_genetic / num_computation)
    correct_genetic.append(cnt_genetic * 100 / num_computation)


plt.plot(N, time_hill, 'r', label='Hill climbing')
plt.plot(N, time_genetic, 'g-.', label='Genetic Algorithm')
# plt.plot(N, time_hill, 'r', label='Genetic Algorithm - population: 20, max_iter: 50')
# plt.plot(N, time_genetic, 'g-.', label='Genetic Algorithm - population: 10, max_iter: 20')

plt.xlabel('Array Size')
plt.ylabel('Time(ms)')

plt.title("Queens' problem")

plt.legend()

plt.show()

plt.plot(N, correct_hill, 'r', label='Hill climbing')
plt.plot(N, correct_genetic, 'g-.', label='Genetic Algorithm')
# plt.plot(N, correct_hill, 'r', label='Genetic Algorithm - population: 20, max_iter: 50')
# plt.plot(N, correct_genetic, 'g-.', label='Genetic Algorithm - population: 10, max_iter: 20')


plt.xlabel('Array Size')
plt.ylabel('Correctness (%)')

plt.title("Queens' problem")

plt.legend()

plt.show()



















