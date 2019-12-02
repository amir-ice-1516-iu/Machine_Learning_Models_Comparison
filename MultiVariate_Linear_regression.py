#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:33:42 2019

@author: rango
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#import seaborn as sb

dataFrame = pd.read_csv("Loan_training.csv")
MeanValue = np.mean(dataFrame.LoanAmount)
dataFrame.LoanAmount.fillna(MeanValue,inplace = True)
#dataFrame.head() # After replacing missing values
newDataFrame = dataFrame.dropna()
def outliers_z_score(ys):
    threshold = 3
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)
def outliers_modified_z_score(ys):
    threshold = 3.5
    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.where(np.abs(modified_z_scores) > threshold)
print("Applying z-score to our dataset")
k = np.array(outliers_z_score(dataFrame.ApplicantIncome))
# k.reshape(len(k[0]))
print("Before removing outliers datashape is ->",dataFrame.shape)
CleandataFrame = dataFrame.drop(k[0])
print("After removing outliers datashape is ->",CleandataFrame.shape)
print("Applying modified z-score to our dataset")
k = np.array(outliers_modified_z_score(dataFrame.ApplicantIncome))
# k.reshape(len(k[0]))
print("Before removing outliers datashape is ->",dataFrame.shape)
CleandataFrame = dataFrame.drop(k[0])
print("After removing outliers datashape is ->",CleandataFrame.shape)
print("Applying z-score to our dataset")
k = np.array(outliers_z_score(dataFrame.ApplicantIncome))
# k.reshape(len(k[0]))
print("Before removing outliers datashape is ->",dataFrame.shape)
CleandataFrame = dataFrame.drop(k[0])
print("After removing outliers datashape is ->",CleandataFrame.shape)
print("Applying modified z-score to our dataset")
k = np.array(outliers_modified_z_score(dataFrame.ApplicantIncome))
# k.reshape(len(k[0]))
print("Before removing outliers datashape is ->",dataFrame.shape)
CleandataFrame = dataFrame.drop(k[0])
print("After removing outliers datashape is ->",CleandataFrame.shape)

#Univariate linear regression
CleandataFrame.to_csv("CleanedData.csv")
X = CleandataFrame.ApplicantIncome
Y = CleandataFrame.LoanAmount
miuX = np.mean(X)
newX = X-miuX
miuY = np.mean(Y)
newY = Y-miuY
k = newX.T @ newX
m = ((newX.T@newY)/k)
yPredicted = m*newX
plt.plot(newX,newY,'g.')
plt.plot(newX,yPredicted,'b*')
error = newY-yPredicted
absoluteError =np.sum(np.fabs(error))/len(newY)
meanSquaredError = np.mean(error**2)#/len(newY)
RootMeanSquaredError = np.sqrt(error.T @ error)/len(newY)
print("Absolute Error = %0.5f" %absoluteError)
print("MeanSquaredError = %0.5f " %meanSquaredError)
print("RootMeanSquaredError = %0.5f" %RootMeanSquaredError)
plt.xlabel("Applicant Income")
plt.ylabel("LoanAmount")
plt.show()

# Multivariate Linear regression
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X1 = np.asarray(CleandataFrame.ApplicantIncome)
X2 = np.asarray(CleandataFrame.CoapplicantIncome)
Idx = X1.argsort()
X1 = X1[Idx]
X2 = X2[Idx]

x1 = X1.reshape((len(X1), 1))
x2 = X2.reshape((len(X2), 1))
Y = np.asarray(CleandataFrame.LoanAmount)
Y = Y[Idx]
y = Y.reshape((len(Y),1))
X = np.append(x1,x2,axis=1)
miuX = np.mean(X, axis=0)
miuY = np.mean(y, axis=0)
ThetaOne = np.linalg.inv(X.T @ X)@X.T@y
ThetaZero = miuY - miuX @ ThetaOne
# print(ThetaZero.shape)
# print(ThetaOne.shape)
def predict(x):
    y_hat = ThetaZero + x @ ThetaOne
    return y_hat[:,0]

ax.plot(X1, X2, Y,'.')
ax.plot(X1, X2, predict(X),'g*')

def f(x,y):
    k = x.shape[0]
    m = x.shape[1]
    x = x.reshape((k*m,1))
    y = y.reshape((k*m,1))
    X = np.append(x,y,axis=1)
    print(X.shape)
    Z = predict(X)
    Z = Z.reshape(k,m)
    print(Z.shape)
    return Z
    

h = np.arange(-2,2,0.1)
print(h.shape)
mx1 = np.min(x1)
mx2 = np.max(x1)
d = mx2-mx1
Mx1 = np.min(x2)
Mx2 = np.max(x2)
D = Mx2-Mx1
xx1 = np.arange(mx1, mx2, 1440)
xx2 = np.arange(Mx1, Mx2, 1440)
xx1,xx2 = np.meshgrid(xx1,xx2)
#ax.plot_surface(xx1,xx2,f(xx1,xx2), rstride=1, cstride=1, cmap=plt.cm.hot)

ax.set_xlabel('ApplicantIncome')
ax.set_ylabel('CoapplicantIncome')
ax.set_zlabel('LoanAmount')
ax.view_init(elev=30,azim=125)
plt.show()