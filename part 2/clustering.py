#-------------------------------------------------------------------------
# AUTHOR: Shane Dirksen
# FILENAME: clustering.py
# SPECIFICATION: A simple program that uses training/testing csv and grid search to find the best k to cluster the data
# FOR: CS 4210- Assignment #5
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix

X_training = df.iloc[:, 0:-1]
y_training = df.iloc[:, -1]

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
scores = []
ks = []
high = 0
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)
     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
    temp = silhouette_score(X_training, kmeans.labels_)
    if (temp > high):
        print("highest k so far:", k)
        high = temp
        kmeans2 = kmeans
    scores.append(silhouette_score(X_training, kmeans.labels_))
    ks.append(k)
#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.plot(ks, scores)
plt.show()
#reading the test data (clusters) by using Pandas library
#--> add your Python code here
df2 = pd.read_csv('testing_data.csv', header=None)

labels = np.array(df2.values).reshape(1, len(df2))[0]
#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
print("Best k", high)
print("Highest silhouette score", max(scores))
print("K-Means Homogeneity Score using the best k = " + metrics.homogeneity_score(labels, kmeans2.labels_).__str__())