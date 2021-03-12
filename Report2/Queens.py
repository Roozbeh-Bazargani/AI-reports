import numpy as np
import random
import time
import matplotlib.pyplot as plt

# N = 8
# A = np.zeros((N, N))
A = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])

def column(A, y):
    if np.sum(A.T[y]):
        return True
    return False

def row(A,x):
    if np.sum(A[x]):
        return True
    return False


def diag(A, x, y):
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


def collision(A, x, y):
    return column(A, y) or row(A, x) or diag(A, x, y)

def IsAnswer(A):
    n = A.shape[0]
    if np.sum(A) != n:
        return False
    for i in range(n):
        for j in range(n):
            if A[i][j] == 1:
                A[i][j] = 0
                if collision(A, i, j):
                    A[i][j] = 1
                    return False
                A[i][j] = 1
    return True

# Solvers ------------------------------------------------------------
def row_random_Queens(N, iter):
    eye = np.eye(N)
    A = eye
    for i in range(iter):
        A = np.random.permutation(eye)
        if IsAnswer(A):
            return A
    print('Could not find any answer!')
    return None

def random_Queens(N, iter):
    A = np.zeros((N,N))
    for i in range(iter):
        rands = random.sample(range(N**2), N)
        for rand in rands:
            x = int(rand/N)
            y = rand%N
            A[x][y] = 1
        if IsAnswer(A):
            return A
        A = np.zeros((N,N))
    print('Could not find any answer!')
    return None


def backtracking_Queens(N):
    A = np.zeros((N,N))
    col = [False]*N
    diag1 = [False] * (2*N - 1)
    diag2 = [False] * (2*N - 1)
    stack = []
    backtracking_search(A, N, 0, col, diag1, diag2, stack)
    while len(stack) > 0:
        x, y = stack.pop()
        A[x][y] = 1
    return A

def backtracking_search(A, N, y, col, diag1, diag2, stack):
    if y == N:
        return stack
    for x in range(N):
        if col[x] or diag1[x+y] or diag2[x-y+N-1]:
            continue
        stack.append([x,y])
        col[x] = diag1[x + y] = diag2[x - y + N - 1] = True
        result = backtracking_search(A, N, y + 1, col, diag1, diag2, stack)
        if result != None:
            if len(result) == N:
              return result
        col[x] = diag1[x + y] = diag2[x - y + N - 1] = False
        stack.pop()


# print(row_random_Queens(10,10000))
# print(random_Queens(8, 100000000))
# print(backtracking_Queens(10))

# main function--------------------------------------------------------
time_random = []
time_row_random = []
time_backtracking = []
A_random = []
A_row_random = []
A_backtracking = []

n = range(4,14)

for i in n:
    tik = time.time()
    temp = (row_random_Queens(i, 1000000))
    tok = time.time()
    # if temp != None:
    A_row_random.append(temp)
    time_row_random.append((tok - tik)*1000)


    tik = time.time()
    temp = (random_Queens(i, 1000000))
    tok = time.time()
    # if temp != None:
    A_random.append(temp)
    time_random.append((tok - tik) * 1000)


    tik = time.time()
    temp = (backtracking_Queens(i))
    tok = time.time()
    A_backtracking.append(temp)
    time_backtracking.append((tok - tik) * 10**3)


print(time_backtracking)
plt.plot(n, time_row_random, 'r', label='Row random')
# plt.plot(n, time_random, 'b:', label='Random')
plt.plot(n, time_backtracking, 'g-.', label='Backtracking')

plt.xlabel('Array Size')
plt.ylabel('Time(ms)')

plt.title("Queens' problem")

plt.legend()

plt.show()

