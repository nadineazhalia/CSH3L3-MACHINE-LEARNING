##  Nama  : Nadine Azhalia Purbani     
##  NIM   : 1301154519
##  Kelas : IF 39-01

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import math
from sklearn import svm

# fungsi untuk memvisualkan data
def visualisasi(data):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for row in data:
        xs = float(row[0])
        ys = float(row[1])
        zs = float(row[2])
        if row[3] == '0':
            c = 'm'
            marker = 'o'
            scatter1 = ax.scatter(xs, ys, zs, c=c, marker=marker)
        elif row[3] == '1':
            c = 'r'
            marker = '^'
            scatter2 = ax.scatter(xs, ys, zs, c=c, marker=marker)
        else:
            c = 'b'
            marker = 's'
            scatter3 = ax.scatter(xs, ys, zs, c=c, marker=marker)
        #print (row)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend([scatter1, scatter2, scatter3], ['Kelas 0', 'Kelas 1', 'Kelas 2'])

    plt.legend()
    plt.show()

# fungsi untuk menghitung nilai g(x) pdf (step pattern layer)
def pdf_count(x1, x2, x3, sigma, x1_train, x2_train, x3_train):
    g = math.exp(-1*((float(x1) - float(x1_train)) ** 2 + (float(x2) - float(x2_train)) **
                   2 + (float(x3) - float(x3_train)) ** 2 / (2 * (sigma ** 2))))
    return g

# fungsi untuk menjumlahkan nilai pdf per-kelas f(x) (step summation layer)
def sum_pdf(data_train, sigma, x_test):
    sum_nol = 0
    sum_satu = 0
    sum_dua = 0
    count_nol = 0
    count_satu = 0
    count_dua = 0
    for row in data_train:
        if row[3] == '0':
            sum_nol += pdf_count(row[0], row[1], row[2],
                                 sigma, x_test[0], x_test[1], x_test[2])
            count_nol += 1

        elif row[3] == '1':
            sum_satu += pdf_count(row[0], row[1], row[2],
                                  sigma, x_test[0], x_test[1], x_test[2])
            count_satu += 1

        else:
            sum_dua += pdf_count(row[0], row[1], row[2],
                                 sigma, x_test[0], x_test[1], x_test[2])
            count_dua += 1
            

    sum_nol = sum_nol / count_nol
    sum_satu = sum_satu / count_satu
    sum_dua = sum_dua / count_dua
    return [sum_nol, sum_satu, sum_dua]

# fungsi untuk mengklasifikasi hasil dari sum_pdf agar dapat menentukan kelas (output layer)
def klasifikasi(data_train, sigma, data_test):
    hasil_y = []
    for row in data_test:
        hasil_pdf = sum_pdf(data_train, sigma, row)
        print(hasil_pdf)
        if hasil_pdf[0] > hasil_pdf[1] and hasil_pdf[0] > hasil_pdf[2]:
            hasil_y.append(0)
        elif hasil_pdf[1] > hasil_pdf[0] and hasil_pdf[1] > hasil_pdf[2]:
            hasil_y.append(1)
        else:
            hasil_y.append(2)
    return hasil_y

# membaca file csv
f_train = open('data_train_PNN.csv', 'r')
f_test = open('data_test_PNN.csv', 'r')
data_train = list(csv.reader(f_train))
data_test = list(csv.reader(f_test))
train_set = data_train[:100]
test_set = data_train[100:]

# memvisualisasika data_train
visualisasi(data_train)

# mengobservasi nilai sigma agar mendapat hasil yg optimal
y_hasil = klasifikasi(data_train, 0.8, data_test)
print (y_hasil)

# memasukkan nilai y_hasil ke list data_test
for i in range(len(data_test)):
    data_test[i].append(str(y_hasil[i]))
# print (data_test)

# memvisualisasikan data_test
# visualisasi(data_test)
f_train.close()

with open ('prediksi.txt', 'w') as file:
    for item in y_hasil:
        print(item, file=file)


# SVM
X = [[0,0], [1,1], [2,2]]
y = [0, 1, 2]
clf = svm.SVC()
clf.fit(X, y)
print (clf.predict([[2., 2.]]))
# print(test_set)


