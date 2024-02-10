import numpy as np

x, y, z = map(int, input().split())
matrix1 = []
matrix2 = []

for i in range(x):
    matrix1.append(list(map(int, input().split())))

for i in range(y):
    matrix2.append(list(map(int, input().split())))

mm = np.dot(matrix1, matrix2)

for i in range(x):
    for j in range(z):
        print(mm[i][j], end = " ")
    print()