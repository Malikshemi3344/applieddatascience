# Importing the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from math import sqrt
from scipy.optimize import curve_fit

# Reading the World bank Data csv file
dataframe = pd.read_csv('worldbankdata.csv')

# Displaying the DataFrame
print('\n', 'Displaying DataFrame:', '\n')
print(dataframe.head())

# information about DataFrame
print('\n', 'Information:', '\n')
print(dataframe.info())

# Taking two variables from the data and Visualizing the data point
X = dataframe[["Year", "Value"]]
plt.scatter(X["Year"], X["Value"], c="blue")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

# --------------- K-MEAN CLUSTERING ---------------
# number of centriod
K=3

# select random observation as a centriod
Centroids = (X.sample(n=K))
plt.scatter(X["Year"], X["Value"], c="blue")
plt.scatter(Centroids["Year"], Centroids["Value"], c="red")
plt.title('Selecting random data-points as a centroid')
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

# Displaying Centroids
print('\n', f'{K} centroids:', '\n')
print(Centroids)


# Step 3 - Assign all the points to the closest cluster centroid
# Step 4 - Recompute centroids of newly formed clusters
# Step 5 - Repeat step 3 and 4
diff = 1
j=0
while(diff!=0):
    XD=X
    i=1
    for index1, row_c in Centroids.iterrows():
        ED=[]
        for index2, row_d in XD.iterrows():
            d1 = (row_c["Year"]-row_d["Year"])**2
            d2 = (row_c["Value"]-row_d["Value"])**2
            d = sqrt(d1+d2)
            ED.append(d)
        X[i] = ED
        i = i+1
    C = []
    for index, row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos = i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Value", "Year"]]
    if j == 0:
        diff = 1
        j = j+1
    else:
        diff = (Centroids_new['Value'] - Centroids['Value']).sum() + (Centroids_new['Year'] - Centroids['Year']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Value","Year"]]

# Final plot of clustered data
color=['blue','green','cyan']
for k in range(K):
    dataframe=X[X["Cluster"]==k+1]
    plt.scatter(dataframe["Year"],dataframe["Value"],c=color[k])
plt.scatter(Centroids["Year"],Centroids["Value"],c='red')
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()