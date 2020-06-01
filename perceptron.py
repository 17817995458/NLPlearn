# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:53:21 2020

@author: 10333
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
#df.label.value_counts()
#plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
#plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
#plt.xlabel('sepal length')
#plt.ylabel('sepal width')
#plt.legend()



class Perceptron:
    def __init__(self):
        self.w=0
        self.b=0
        self.rate=0.1 #学习速率
        
    def sign(self,w,x,b):
        return np.dot(w,x)+b
    
    def fit(self,X_train,Y_train):
        self.w=np.ones(len(X_train[0]))
        wrong=True
        j=0
        while wrong:
            cnt=0
            print(j)
            j+=1
            for i in range(len(Y_train)):
                x=X_train[i]
                y=Y_train[i]
                if self.sign(self.w,x,self.b)*y <=0 :
                    self.w = self.w + self.rate*x*y
                    self.b = self.b + self.rate*y
                    cnt+=1
            if cnt == 0:
                wrong = False
                
     
class PerceptronDual:
    def __init__(self):
        self.alfa=0
        self.b=0
        self.w=0

    def compute(self,x,X_train,Y_train):
        sum1=sum(np.dot(X_train[j]*Y_train[j]*self.alfa[j],x) for j in range(len(X_train)))
        return sum1+self.b

    def fit(self,X_train,Y_train):
        self.w=np.ones(len(X_train[0]))
        self.alfa=np.zeros(len(X_train))
        wrong=True
        while wrong:
            cnt=0
            for i in range(len(Y_train)):
                x=X_train[i]
                y=Y_train[i]
                if self.compute(x,X_train,Y_train)*y<=0 :
                    self.alfa[i] += 1 
                    self.b = self.b + y
                    cnt+=1
            if cnt == 0:
                wrong = False
                
        self.w=sum(self.alfa[j]*Y_train[j]*X_train[j] for j in range(len(Y_train)))

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])       
        
perceptron = PerceptronDual()
perceptron.fit(X, y)
x_points = np.linspace(4, 7,10)
y_ = -(perceptron.w[0]*x_points + perceptron.b)/perceptron.w[1]
plt.plot(x_points, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='-1')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
    


           