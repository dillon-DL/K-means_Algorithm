# Firstly we will be reding the excel files files in which the data is stored
# The program will then iterate through the information inorder to do analysis
# It will thenuse the k-means algorithm to cluster the data into groups which are closely related
# Lastly the data is then displayed via a grapgh for visualisation of the data in a user friendly manner as well as the names of all the countries and the cluster they fall under

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
from scipy.spatial import distance
import random
import csv



def read_data_file(fileAddress, column1, column2):
    csv_file = pd.read_csv(fileAddress)
    xy_data = csv_file[[column1, column2]]
    dataList = xy_data.astype(float).values.tolist()

    return dataList



def get_countries(file_address):
    csv_file = pd.read_csv(file_address)
    countries_column = csv_file[['Countries']]
    countries_list = countries_column.astype(str).values.tolist()

    return countries_list



fileRead = 0

while fileRead < 1 or fileRead > 3:
    fileRead = int(input("Enter the number which corresponds to the data you want to retrieve below:\n"
                             "1. Data from the year 1953\n"
                             "2. Data from the year 2008\n"
                             "3. Data from both year 1953 and 2008\n"))

    if fileRead == 1:
        data_list = read_data_file(r"data1953.csv", 'BirthRate(Per1000 - 1953)', 'LifeExpectancy(1953)')
        countries_list = get_countries(r"data1953.csv")

    elif fileRead == 2:
        data_list = read_data_file(r"data2008.csv", 'BirthRate(Per1000 - 2008)', 'LifeExpectancy(2008)')
        countries_list = get_countries(r"data2008.csv")

    elif fileRead == 3:
        data_list = read_data_file(r"dataBoth.csv", 'BirthRate(Per1000)', 'LifeExpectancy')
        countries_list = get_countries(r"dataBoth.csv")

    else:
        print("Invalid option.")


number_of_clusters = int(input("Number of clusters: "))
n_of_iterations = int(input("Number of iterations: "))


K_means = KMeans(n_clusters=number_of_clusters, max_iter=n_of_iterations)
K_means.fit(data_list)
centroids = K_means.cluster_centers_
cluster_name = K_means.labels_
dist = K_means.inertia_


colours = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'w.']

for i in colours:

    while 2 > len(colours):
        colours.append(i)

data_clustered = []
values = []

for i in range(len(data_list)):
    data_clustered.append([countries_list[i], data_list[i], str(cluster_name[i])])
    values.append([data_list[i], str(cluster_name[i])])
    plt.plot(data_list[i][0], data_list[i][1], colours[cluster_name[i]], markersize=10)


plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", alpha = 0.3)
plt.xlabel('BirthRate')
plt.ylabel('Life Expectancy')
plt.show()


cluster_list = []

for i in range(number_of_clusters):
    cluster_list.append(str(i))

cluster_for_values = {key: [] for key in cluster_list}
data_values = cluster_for_values.items()

for i in data_values:

    for j in values:
        if i[0] == j[1]:
            i[1].append(j[0])


cluster_dict = {key: [] for key in cluster_list} 
cluster_data = cluster_dict.items()  

for i in cluster_data:

    for j in data_clustered:
        if i[0] == j[2]:
            i[1].append(j[0:2])

print("\n\nAMOUNT OF COUNTRIES USED:\n")

for i in cluster_data:
    print("There are", (len(i[1])), "countries in the cluster", (i[0]))


for key, value in cluster_data:
    print("\n\nCOUNTRIES IN CLUSTER:", (key[0]), ":\n")

    for list_ in value:
        print(list_[0])


print("\n\nAVERAGES for BIRTH RATE AND LIFE EXPECTANCY:\n")

x_birth = []
y_life = []

for key, value in data_values:

    for list_ in value:
        x_birth.append(list_[0])
        y_life.append(list_[1])

    meanX = np.mean(x_birth)
    print("cluster", (key), ":")
    print("Birth rate:", (round(meanX, 2)))

    meansY = np.mean(y_life)
    print("Life expectancy:", (round(meansY, 2)), "\n")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
from scipy.spatial import distance
import random
import csv



def read_data_file(fileAddress, column1, column2):
    csv_file = pd.read_csv(fileAddress)
    xy_data = csv_file[[column1, column2]]
    dataList = xy_data.astype(float).values.tolist()

    return dataList



def get_countries(file_address):
    csv_file = pd.read_csv(file_address)
    countries_column = csv_file[['Countries']]
    countries_list = countries_column.astype(str).values.tolist()

    return countries_list



fileRead = 0

while fileRead < 1 or fileRead > 3:
    fileRead = int(input("Enter the number which corresponds to the data you want to retrieve below: \n\n"
                             "1. Data from the year 1953\n"
                             "2. Data from the year 2008\n"
                             "3. Data from both year 1953 and 2008\n"))

    if fileRead == 1:
        data_list = read_data_file(r"data1953.csv", 'BirthRate(Per1000 - 1953)', 'LifeExpectancy(1953)')
        countries_list = get_countries(r"data1953.csv")

    elif fileRead == 2:
        data_list = read_data_file(r"data2008.csv", 'BirthRate(Per1000 - 2008)', 'LifeExpectancy(2008)')
        countries_list = get_countries(r"data2008.csv")

    elif fileRead == 3:
        data_list = read_data_file(r"dataBoth.csv", 'BirthRate(Per1000)', 'LifeExpectancy')
        countries_list = get_countries(r"dataBoth.csv")

    else:
        print("Invalid option.")


number_of_clusters = int(input("Number of clusters: "))
n_of_iterations = int(input("Number of iterations: "))


K_means = KMeans(n_clusters=number_of_clusters, max_iter=n_of_iterations)
K_means.fit(data_list)
centroids = K_means.cluster_centers_
cluster_name = K_means.labels_
dist = K_means.inertia_


colours = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'w.']

for i in colours:

    while 2 > len(colours):
        colours.append(i)

data_clustered = []
values = []

for i in range(len(data_list)):
    data_clustered.append([countries_list[i], data_list[i], str(cluster_name[i])])
    values.append([data_list[i], str(cluster_name[i])])
    plt.plot(data_list[i][0], data_list[i][1], colours[cluster_name[i]], markersize=10)


plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", alpha = 0.3)
plt.xlabel('BirthRate')
plt.ylabel('Life Expectancy')
plt.show()


cluster_list = []

for i in range(number_of_clusters):
    cluster_list.append(str(i))

cluster_for_values = {key: [] for key in cluster_list}
data_values = cluster_for_values.items()

for i in data_values:

    for j in values:
        if i[0] == j[1]:
            i[1].append(j[0])


cluster_dict = {key: [] for key in cluster_list}  
cluster_data = cluster_dict.items() 

for i in cluster_data:

    for j in data_clustered:
        if i[0] == j[2]:
            i[1].append(j[0:2])


print("\n\nAMOUNT OF COUNTRIES USED:\n")

for i in cluster_data:
    print("There are", (len(i[1])), "countries in the cluster", (i[0]))

for key, value in cluster_data:
    print("\n\nCOUNTRIES IN CLUSTER:", (key[0]), ":\n")

    for list_ in value:
        print(list_[0])


print("\n\nAVERAGES for BIRTH RATE AND LIFE EXPECTANCY:\n")

x_birth = []
y_life = []

for key, value in data_values:

    for list_ in value:
        x_birth.append(list_[0])
        y_life.append(list_[1])

    meanX = np.mean(x_birth)
    print("Cluster", (key), ":")
    print("Birth rate:", (round(meanX, 2)))

    meansY = np.mean(y_life)
    print("Life expectancy:", (round(meansY, 2)), "\n")
