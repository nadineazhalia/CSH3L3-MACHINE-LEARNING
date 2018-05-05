# Nama  : Nadine Azhalia
# NIM   : 1301154519
# Kelas : IF 39-01

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

# membaca file txt
file = open('TestsetTugas2.txt', "r")
lines = file.read().split("\n")

X = ["" for x in range(len(lines))]
Y = ["" for x in range(len(lines))]

i = 0
for l in lines:
    values = l.split("\t")
    X[i] = values[0]
    Y[i] = values[1]
    i += 1

X = [float(s) for s in X]
Y = [float(s) for s in Y]

data = []
for i in range(len(X)):
    data.append([X[i], Y[i]])

data = np.array([X,Y]).T
# visualisasi data
plt.scatter(data[:,0], data[:,1], c='blue', s=7)

# fungsi menghitung nilai euclide
def distc(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# menentukan jumlah K dan me-random nilai centroid
k = 5
centroid = np.random.rand(k,2) * 36
print ("Nilai Centroid Awal :")
print(centroid)
plt.scatter(centroid[:,0], centroid[:,1], marker='*', c='r')
plt.show()

cent_awal = np.zeros_like(centroid)
clusters = np.zeros(len(data))
error = 1

# perulangan akan dilakukan selama jarak centroid ke cent_awal tidak sama dengan 0
while (error > 0):
    # perulangan untuk menentukan jarak euclid dari centroid awal ke setiap data
    for i in range(len(data)):
        euclide = distc(data[i,:], centroid, ax=1)
        cluster = np.argmin(euclide)
        clusters[i] = cluster
    cent_awal[:,:] = centroid[:,:]

    # perulangan untuk menentukan centroid baru (didapat dari rata-rata data per cluster)
    for i in range(k):
        points = []
        for j in range(len(data)):
            if clusters[j] == i:
                points.append(data[j])
        centroid[i] = np.mean(points, axis=0)
    # print(centroid)
    error = distc(centroid, cent_awal, None)
    # print(error)

print("Nilai Centroid Akhir :")
print(centroid)

# fungsi menentukan nilai dari SSE
# def sse (k, data, centroid):
#     sse = 0    
#     for i in range(k):
#         points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
#         sse += np.sum((centroid[i] - points) ** 2)
#     return sse

# print("Nilai SSE :")
# print(sse(5,data,centroid))

print("Hasil Prediksi:")
print(clusters)   
 
# membuat visualisasi hasil pengelompokan data dengan data centroid yang baru
colors = ['b', 'm', 'r', 'c', 'g']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
    ax.scatter(points[:,0], points[:,1], s=7, c=colors[i])
ax.scatter(centroid[:,0], centroid[:,1], marker='x', s=70, c='k')

plt.show()

# import hasil cluster ke file csv
with open ('HasilCLuster-asdos.csv', 'w') as file:
    for item in clusters:
        print(item, file=file)