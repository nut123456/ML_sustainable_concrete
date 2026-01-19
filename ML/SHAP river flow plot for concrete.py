# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:29:17 2024

@author: PC
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_excel("DATA324_Modeling.xlsx", engine='openpyxl')

# Strip whitespace from column names to ensure consistency
df.columns = df.columns.str.strip()

# Feature columns based on the dataset
feature_columns = [
    'CEMENTE', 'BLAST', 'FLY', 'WATER', 'SUPERPLASTICIZER', 'COARSE', 'FINE'
]

# Define the target column
target_column = 'CONCRETE'

# Ensure the specified columns are present in the DataFrame
df = df[feature_columns + [target_column]]

# Plotting
plt.figure(figsize=(10, 6))

# Create a color palette based on the target values
norm = plt.Normalize(df[target_column].min(), df[target_column].max())
colors = plt.cm.plasma(norm(df[target_column]))

# Plot each row of data with color corresponding to the target value
for i, row in df.iterrows():
    plt.plot(feature_columns, row[feature_columns], color=colors[i], alpha=0.3)

# Expected value line (assuming 42.92 MPa based on your graph)
expected_value = 42.92
plt.plot(feature_columns, [expected_value] * len(feature_columns), color='black', linestyle='--', label=f'Expected value: {expected_value} MPa')

# Colorbar for the target variable
sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
sm.set_array([])  # Only needed for colorbar
cbar = plt.colorbar(sm)
cbar.set_label('Mean compressive strength of concrete (MPa)')

# Customize plot appearance
plt.xticks(rotation=90)
plt.xlabel("Properties")
plt.ylabel("Predicted output compressive strength (MPa)")
plt.legend()
plt.tight_layout()
plt.show() 