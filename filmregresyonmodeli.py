# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 23:32:24 2023

@author: Baris
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_excel('filmverisetisayisalversiyon.xlsx')

df.dropna(subset=['Distributor'], inplace=True)

sbn.scatterplot(x="Distributor",y="Gross",data=df)
plt.figure(figsize=(7,5))
sbn.scatterplot(df["Gross"])
plt.show()


df=df.sort_values("Gross", ascending=False).iloc[90:]
df = df.sort_values('Gross', ascending=True).iloc[90:]

gross_mean2 = df['Gross'].mean()

print(f"Gross Değerlerinin Ortalaması: {gross_mean2}")
print(" ")
print(" ")
print(" ")

sbn.scatterplot(x="Distributor",y="Gross",data=df)
plt.figure(figsize=(7,5))
sbn.scatterplot(df["Gross"])
plt.show()

X=df.drop("Gross", axis=1)
y=df["Gross"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#decision tree

model = DecisionTreeRegressor()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

sonuclar = {
    "MAE": mae,
    'RMSE': rmse,
    'R2': r2
    }

print("Decision Tree Regression Results:")
print(sonuclar)
plt.scatter(y_test,y_pred)
plt.plot(y_test,y_test, "g-*")
plt.show()

#knn

knn_model = KNeighborsRegressor()

knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)

mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)

sonuclar_knn = {
    "MAE": mae_knn,
    'RMSE': rmse_knn,
    'R2': r2_knn
}

print("KNN Regression Results:")
print(sonuclar_knn)
plt.scatter(y_test,y_pred)
plt.plot(y_test,y_test, "g-*")
plt.show()

#lineer regresyon
linear_model_unscaled = LinearRegression()
linear_model_unscaled.fit(X_train, y_train)
y_linear_pred_unscaled = linear_model_unscaled.predict(X_test)

# Scaling uygulanmış veriler
linear_model_scaled = LinearRegression()
linear_model_scaled.fit(X_train_scaled, y_train)
y_linear_pred_scaled = linear_model_scaled.predict(X_test_scaled)


linear_metrics = {
    "MAE": mean_absolute_error(y_test, y_linear_pred_scaled),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_linear_pred_scaled)),
    'R2': r2_score(y_test, y_linear_pred_scaled)
}
print("Linear Regression Results:")
print(linear_metrics)
plt.scatter(y_test,y_linear_pred_scaled)
plt.plot(y_test,y_test, "g-*")
plt.show()


# Polinom Regresyon
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_poly_pred = poly_model.predict(X_test_poly)
poly_metrics = {
    "MAE": mean_absolute_error(y_test, y_poly_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_poly_pred)),
    'R2': r2_score(y_test, y_poly_pred)
}
print("Polynomial Regression Results:")
print(poly_metrics)
plt.scatter(y_test,y_poly_pred)
plt.plot(y_test,y_test, "g-*")
plt.show()

#Ridge
ridge_model = Ridge()
ridge_model.fit(X_train_scaled, y_train)
y_ridge_pred = ridge_model.predict(X_test_scaled)
ridge_metrics = {
    "MAE": mean_absolute_error(y_test, y_ridge_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_ridge_pred)),
    'R2': r2_score(y_test, y_ridge_pred)
}
print("Ridge Regression Results:")
print(ridge_metrics)
plt.scatter(y_test,y_ridge_pred)
plt.plot(y_test,y_test, "g-*")
plt.show()

# Lasso Regresyon
lasso_model = Lasso()
lasso_model.fit(X_train_scaled, y_train)
y_lasso_pred = lasso_model.predict(X_test_scaled)
lasso_metrics = {
    "MAE": mean_absolute_error(y_test, y_lasso_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_lasso_pred)),
    'R2': r2_score(y_test, y_lasso_pred)
}
print("Lasso Regression Results:")
print(lasso_metrics)
plt.scatter(y_test,y_lasso_pred)
plt.plot(y_test,y_test, "g-*")
plt.show()

# Elastic Net Regresyon
elastic_net_model = ElasticNet()
elastic_net_model.fit(X_train_scaled, y_train)
y_elastic_pred = elastic_net_model.predict(X_test_scaled)
elastic_metrics = {
    "MAE": mean_absolute_error(y_test, y_elastic_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_elastic_pred)),
    'R2': r2_score(y_test, y_elastic_pred)
}
print("Elastic Net Regression Results:")
print(elastic_metrics)
plt.scatter(y_test,y_elastic_pred)
plt.plot(y_test,y_test, "g-*")
plt.show()

# Support Vector Regression
svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)
y_svr_pred = svr_model.predict(X_test_scaled)
svr_metrics = {
    "MAE": mean_absolute_error(y_test, y_svr_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_svr_pred)),
    'R2': r2_score(y_test, y_svr_pred)
}
print("Support Vector Regression Results:")
print(svr_metrics)
plt.scatter(y_test,y_svr_pred)
plt.plot(y_test,y_test, "g-*")
plt.show()

# Rastgele Orman Regresyon
rf_model = RandomForestRegressor()
rf_model.fit(X_train_scaled, y_train)
y_rf_pred = rf_model.predict(X_test_scaled)
rf_metrics = {
    "MAE": mean_absolute_error(y_test, y_rf_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_rf_pred)),
    'R2': r2_score(y_test, y_rf_pred)
}
print("Random Forest Regression Results:")
print(rf_metrics)
print(df.iloc[1500])
yeniSeries = df.drop("Gross",axis=1).iloc[1500]
yeniSeries = scaler.transform(yeniSeries.values.reshape(-1,2))
print(rf_model.predict(yeniSeries))
plt.scatter(y_test,y_rf_pred)
plt.plot(y_test,y_test, "g-*")
plt.show()

#Poisson
poisson_model = PoissonRegressor()
poisson_model.fit(X_train_scaled, y_train)
y_poisson_pred = poisson_model.predict(X_test_scaled)
poisson_metrics = {
    "MAE": mean_absolute_error(y_test, y_poisson_pred),
    'RNSE': np.sqrt(mean_squared_error(y_test, y_poisson_pred)),
    'R2': r2_score(y_test, y_poisson_pred)
}
print("Poisson Regression Results:")
print(poisson_metrics)
plt.scatter(y_test,y_poisson_pred)
plt.plot(y_test,y_test, "g-*")
plt.show()

