# -*- coding: utf-8 -*-
"""
Created on Tue Apr 7 12:52:44 2020

@author: Esra
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriseti=pd.read_csv('IceCreamData.csv')
veriseti.head()

X=veriseti['Temperature'].values
y=veriseti['Revenue'].values
uzunluk=len(X)
X=X.reshape((uzunluk,1))

#%%

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#%%
from sklearn.linear_model import LinearRegression
model_Regression=LinearRegression()
model_Regression.fit(X_train,y_train)

y_pred= model_Regression.predict(X_test)
print("tahmini gelir: ",model_Regression.predict([[30]]))
#%% Eğitim seti sonuçlarının grafiğinin çizilmesi
plt.scatter(X_train,y_train,color="red")
plt.xlabel("sıcaklık")
plt.ylabel("gelir")
plt.plot(X_train,model_Regression.predict(X_train),color="blue")
plt.show()

#%% Test seti sonuçlarının grafiğinin çizilmesi
plt.scatter(X_test,y_test,color="red")
plt.xlabel("sıcaklık")
plt.ylabel("gelir")
plt.plot(X_train,model_Regression.predict(X_train),color="blue")
plt.show()

#%%
print('eğim(Q1):',model_Regression.coef_)
print('kesen(Q0):',model_Regression.intercept_)
print("y=%0.2f"%model_Regression.coef_+"x+%0.2f"%model_Regression.intercept_)

#%% decision tree regression

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X,y)

y_head = tree_reg.predict(X)

X_=np.arange(min(X),max(X),0.01).reshape(-1,1)
y_head = tree_reg.predict(X_)

plt.scatter(X,y,color="red")
plt.plot(X_,y_head,color="green")
plt.xlabel("sıcaklık")
plt.ylabel("gelir")
plt.show()
print("tahmini gelir2: ",tree_reg.predict([[32]]))
#%% random forest regression

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100,random_state=42)

rf.fit(X,y)
y_head_orj = rf.predict(X)

x_=np.arange(min(X),max(X),0.01).reshape(-1,1)
y_head = rf.predict(X_)

plt.scatter(X,y,color="red")
plt.plot(X_,y_head,color="blue")
plt.xlabel("sıcaklık")
plt.ylabel("gelir")
plt.show()
print("tahmini gelir3: ",rf.predict([[30]]))
#%%
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score
print("r2: ",r2_score(y_test,y_pred))
print("MAE: ",mean_absolute_error(y_test,y_pred))
print("MSE: ",mean_squared_error(y_test,y_pred))
print("MAPE: ",median_absolute_error(y_test,y_pred))