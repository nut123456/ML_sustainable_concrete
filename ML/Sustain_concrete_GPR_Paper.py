# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 22:39:04 2025

@author: Admin
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Reload the dataset since previous cell context was lost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Import RBF kernel and redefine the optimized kernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic

# Load dataset
df = pd.read_csv('concrete_data.csv')

# Define features and target again
X = df.drop(columns=["concrete_compressive_strength"])
y = df["concrete_compressive_strength"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define a base kernel structure
base_kernel = ConstantKernel() * (RationalQuadratic() + RBF())

# Create the model object
gpr = GaussianProcessRegressor(kernel=base_kernel, random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    "alpha": [0.01, 0.1, 1.0],  # Noise level
    "kernel__k1__constant_value": [0.1, 1.0, 10.0, 100.0],  # ConstantKernel
    "kernel__k2__k1__alpha": [0.1, 1.0, 10.0],  # RationalQuadratic alpha
    "kernel__k2__k1__length_scale": [0.1, 1.0, 10.0],  # RationalQuadratic length_scale
    "kernel__k2__k2__length_scale": [0.1, 1.0, 10.0]   # RBF length_scale
}

# Run GridSearchCV (cv=3-fold cross-validation)
grid_search = GridSearchCV(gpr, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model evaluation on test set
best_gpr = grid_search.best_estimator_
y_pred_best = best_gpr.predict(X_test_scaled)

# Final performance
accuracy_best = r2_score(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)

accuracy_best, mae_best, mse_best, rmse_best #accuracy_best=0.9028488861791896
# Create DataFrame for actual and predicted values
plt.figure(dpi=1000)  # Set high DPI for enhanced clarity
comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_best
})

# Create the pairplot
pairplot = sns.pairplot(
    comparison_df,
    kind='reg',
    diag_kind='kde',
    plot_kws={'line_kws': {'color': 'red', 'linewidth': 1.5}, 'scatter_kws': {'s': 25, 'alpha': 0.9}},
    diag_kws={'shade': True, 'linewidth': 2}
)

# Adjusting title and layout
pairplot.fig.suptitle('Pairplot of Actual and Predicted Values', y=1.02, fontsize=16)
pairplot.fig.set_size_inches(10, 10)  # Increase figure size
# Add grid and adjust axis label size
for ax in pairplot.axes.flat:  # Iterate through each subplot
    ax.grid(alpha=0.5)  # Add grid with transparency
    ax.xaxis.set_tick_params(labelsize=16)  # Increase x-axis label size
    ax.yaxis.set_tick_params(labelsize=16)  # Increase y-axis label size
    ax.set_xlabel(ax.get_xlabel(), fontsize=20, labelpad=10)  # Set x-axis label size
    ax.set_ylabel(ax.get_ylabel(), fontsize=20, labelpad=10)  # Set y-axis label size

plt.show()
