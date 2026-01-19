# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:27:50 2024

@author: PC
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, InputLayer

# Define the models including GPR, ANN (MLP), and CNN
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'MLP Regressor': MLPRegressor(random_state=42),
    'Gaussian Process Regressor': GaussianProcessRegressor(kernel=ConstantKernel(1.0, (1e-2, 1e2)) * RationalQuadratic(), alpha=1.0, random_state=0)
}

# Load dataset
df = pd.read_csv('concrete_data.csv', sep=',')
feature_columns = [col for col in df.columns if col not in ["concrete_compressive_strength"]]
target_column = "concrete_compressive_strength"
X = df[feature_columns]
y = df[target_column]

# Splitting the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating a list to store results
results_list = []

# Evaluate sklearn models
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    results_list.append({'Model': name, 'Accuracy': accuracy, 'MAE': mae, 'MSE': mse, 'RMSE': rmse})

# Adding ANN model (using TensorFlow/Keras MLP)
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
ann_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
ann_predictions = ann_model.predict(X_test).flatten()
ann_mae = mean_absolute_error(y_test, ann_predictions)
ann_mse = mean_squared_error(y_test, ann_predictions)
ann_rmse = np.sqrt(ann_mse)
results_list.append({'Model': 'Artificial Neural Network', 'Accuracy': ann_model.evaluate(X_test, y_test, verbose=0)[1], 'MAE': ann_mae, 'MSE': ann_mse, 'RMSE': ann_rmse})

# Adding CNN model
cnn_model = Sequential([
    InputLayer(input_shape=(X_train.shape[1], 1)),
    Conv1D(64, kernel_size=2, activation='relu'),
    Flatten(),
    Dense(1)
])
cnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
cnn_model.fit(X_train[..., np.newaxis], y_train, epochs=50, batch_size=10, verbose=0)
cnn_predictions = cnn_model.predict(X_test[..., np.newaxis]).flatten()
cnn_mae = mean_absolute_error(y_test, cnn_predictions)
cnn_mse = mean_squared_error(y_test, cnn_predictions)
cnn_rmse = np.sqrt(cnn_mse)
results_list.append({'Model': 'Convolutional Neural Network', 'Accuracy': cnn_model.evaluate(X_test[..., np.newaxis], y_test, verbose=0)[1], 'MAE': cnn_mae, 'MSE': cnn_mse, 'RMSE': cnn_rmse})

# Convert results list to DataFrame
results = pd.DataFrame(results_list)
results.sort_values(by='RMSE', ascending=True, inplace=True)
print(results)

