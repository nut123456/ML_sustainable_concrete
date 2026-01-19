# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:27:09 2024

@author: PC
"""


# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Loading the dataset
df = pd.read_csv('concrete_data.csv', sep=',')

# Defining features and target variable
feature_columns = [col for col in df.columns if col not in ["concrete_compressive_strength"]]
target_column = "concrete_compressive_strength"
X = df[feature_columns]
y = df[target_column]

# Splitting the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Defining the Gaussian Process Regressor with Rational Quadratic kernel
kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RationalQuadratic()
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1.0, random_state=25)
gpr.fit(X_train, y_train)

# Make predictions
train_predictions = gpr.predict(X_train)
test_predictions = gpr.predict(X_test)

# Combine training and test data for plotting
total_actual = np.concatenate([y_train, y_test])
total_predicted = np.concatenate([train_predictions, test_predictions])

# Create the plot
plt.figure(figsize=(12, 8), dpi=1500)

# Plot actual and predicted values with hollow circles
plt.scatter(range(len(y_train)), y_train, edgecolors='black', facecolors='none', s=50, label='Actual value', zorder=3.5)  # Hollow circles for training actual values
plt.scatter(range(len(y_train), len(total_actual)), y_test, edgecolors='black', facecolors='none', s=50, zorder=3.5)  # Hollow circles for test actual values
plt.scatter(range(len(total_actual)), total_predicted, color='red', s=40, label='Predicted value', zorder=3.5)
# Add vertical line for training/test split
plt.axvline(x=len(y_train), color='blue', linestyle='--', linewidth=2.5, label='Training(left side)/Test Split (right side)', zorder=1)

# Set axis labels and title
plt.xlabel('Sample number', fontsize=24, fontfamily='Times New Roman')
plt.ylabel('CS:Compressive strength (MPa)', fontsize=24, fontfamily='Times New Roman')
#plt.title('Actual vs Predicted Compressive Strength', fontsize=16, fontfamily='Times New Roman')
# Customize tick labels font size
plt.tick_params(axis='both', which='major', labelsize=18)  # Increase font size for major ticks
plt.tick_params(axis='both', which='minor', labelsize=14)  # Increase font size for minor ticks

# Add legend
plt.legend(fontsize=18)
plt.legend(loc='upper right', fontsize=18)  # Increased font size
# Customize grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()