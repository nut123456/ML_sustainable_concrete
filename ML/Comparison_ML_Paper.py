# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 16:10:01 2025

@author: Admin
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic, RBF

# Set global font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Load dataset
df = pd.read_csv('concrete_data.csv')
X = df.drop(columns=["concrete_compressive_strength"])
y = df["concrete_compressive_strength"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Tuned GPR using GridSearchCV
base_kernel = ConstantKernel() * (RationalQuadratic() + RBF())
gpr_base = GaussianProcessRegressor(kernel=base_kernel, random_state=42)

param_grid = {
    "alpha": [0.01, 0.1, 1.0],
    "kernel__k1__constant_value": [0.1, 1.0, 10.0, 100.0],
    "kernel__k2__k1__alpha": [0.1, 1.0, 10.0],
    "kernel__k2__k1__length_scale": [0.1, 1.0, 10.0],
    "kernel__k2__k2__length_scale": [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(gpr_base, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_gpr = grid_search.best_estimator_

# Define models
models = {
    'Linear': LinearRegression(),
    'Lasso': Lasso(alpha=0.1, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'MLP Regressor': MLPRegressor(random_state=42),
    'GPR': best_gpr
}

# Plot settings
n_models = len(models)
n_cols = 4
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5.5 * n_rows), sharey=True, dpi=1000)
axes = axes.flatten()

# Loop through each model
for idx, (name, model) in enumerate(models.items()):
    use_scaled = name in ['SVR', 'KNN', 'MLP Regressor', 'GPR']
    X_train_use = X_train_scaled if use_scaled else X_train
    X_test_use = X_test_scaled if use_scaled else X_test

    model.fit(X_train_use, y_train)
    y_train_pred = model.predict(X_train_use)
    y_test_pred = model.predict(X_test_use)

    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)

    ax = axes[idx]
    ax.scatter(y_train, y_train_pred, color='blue', s=40, label='Train data')
    ax.scatter(y_test, y_test_pred, facecolors='none', edgecolors='red', s=40, label='Test data')

    lims = [min(y.min(), y_train_pred.min(), y_test_pred.min()), 
            max(y.max(), y_train_pred.max(), y_test_pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=2, label='1:1 Line')
    ax.plot(lims, [v*1.2 for v in lims], 'r--', linewidth=1, label='20%')
    ax.plot(lims, [v*0.8 for v in lims], 'r--', linewidth=1)
    # ax.plot(lims, lims, 'r--', linewidth=1, label='20%')

    
    ax.set_title(name, fontsize=20, fontweight='bold')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Actual CS (MPa)', fontsize=18)
    if idx % n_cols == 0:
        ax.set_ylabel('Predicted CS (MPa)', fontsize=18)

    ax.text(0.05, 0.95,
            f"R² = {r2:.3f}\nMAE = {mae:.2f}\nMSE = {mse:.2f}\nRMSE = {rmse:.2f}",
            transform=ax.transAxes,
            fontsize=18, verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

# Remove unused axes
for j in range(len(models), len(axes)):
    fig.delaxes(axes[j])

# Shared legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', 
           ncol=4, fontsize=24, markerscale=2, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.92])

# Save high-resolution image
plt.savefig("model_comparison.png", dpi=1000, bbox_inches='tight')

plt.show()
