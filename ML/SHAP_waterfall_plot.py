# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 22:40:39 2025

@author: อาณัติ
"""
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

# =========================
# 🎨 1. Style Configuration
# =========================
plt.style.use('default')  # reset
plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 16,
    "text.color": "black",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5
})

# =========================
# ⏱ 2. Start timing
# =========================
start_time = time.time()

# =========================
# 📥 3. Load dataset
# =========================
df = pd.read_csv('concrete_data_SHAP.csv')
X = df.drop(columns=['CS'])
y = df['CS']

# =========================
# ⚖️ 4. Preprocessing
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_sample = pd.DataFrame(X_scaled, columns=X.columns)

# =========================
# 🔀 5. Split data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 🔮 6. Train GPR model
# =========================
kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RationalQuadratic()
model = GaussianProcessRegressor(kernel=kernel, alpha=1.0, random_state=0)
model.fit(X_train, y_train)

# =========================
# 🔍 7. SHAP Explainer
# =========================
background_data = X_sample.sample(100, random_state=0)
explainer = shap.KernelExplainer(model.predict, background_data)

# =========================
# 🎯 8. Select data points
# =========================
min_idx = y.idxmin()
max_idx = y.idxmax()
median_strength = y.median()
mid_idx = (y - median_strength).abs().idxmin()

print(f"Low Strength at index {min_idx}: {y[min_idx]:.2f} MPa")
print(f"Median Strength at index {mid_idx}: {y[mid_idx]:.2f} MPa")
print(f"High Strength at index {max_idx}: {y[max_idx]:.2f} MPa")

# =========================
# 📈 9. Compute SHAP values
# =========================
shap_values_min = explainer.shap_values(X_sample.iloc[[min_idx]])
shap_values_mid = explainer.shap_values(X_sample.iloc[[mid_idx]])
shap_values_max = explainer.shap_values(X_sample.iloc[[max_idx]])

# =========================
# 💧 10. Waterfall plot function
# =========================
def plot_waterfall(shap_values, instance_idx, label):
    shap_exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_sample.iloc[instance_idx].values,
        feature_names=X.columns.tolist()
    )
    fig = plt.figure(dpi=500)
    shap.plots.waterfall(shap_exp, show=False)
    plt.title(f"{label} Strength ({y[instance_idx]:.2f} MPa)", fontsize=16)
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig(f"SHAP_Waterfall_{label}_Strength.png", dpi=500, bbox_inches="tight", facecolor='white')
    plt.show()

# =========================
# 🖼️ 11. Generate plots
# =========================
plot_waterfall(shap_values_min, min_idx, "Low")
plot_waterfall(shap_values_mid, mid_idx, "Median")
plot_waterfall(shap_values_max, max_idx, "High")

# =========================
# ⏱ 12. Elapsed time
# =========================
elapsed_time = time.time() - start_time
print(f"Elapsed Time: {elapsed_time:.2f} seconds") #Elapsed Time: 8.56 seconds
