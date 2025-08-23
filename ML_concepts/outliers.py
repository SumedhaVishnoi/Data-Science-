# outliers distort the data 
# these are the extreme values that deviate a lot from the rest of the data 
# 
# METHODS - Z score method (standard deviation method )
# works for normally distributed data 
# 
# - IQR method (interquartile range method)
# works for skewed data 
# 
# - Boxplot Visualization
#    - Scatter plot visualization

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sample dataset
df = pd.DataFrame({'Salary': [30, 35, 40, 45, 50, 200]})
print("Original Data:\n", df)

# 1. Z-Score Method
mean = df['Salary'].mean()
std = df['Salary'].std()
df['Z_score'] = (df['Salary'] - mean) / std
print("\nWith Z-Scores:\n", df)

# Removing outliers (|Z| > 3)
df_no_outlier_z = df[df['Z_score'].abs() <= 3]
print("\nAfter Removing Outliers (Z-Score):\n", df_no_outlier_z)

# 2. IQR Method
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_no_outlier_iqr = df[(df['Salary'] >= lower) & (df['Salary'] <= upper)]
print("\nAfter Removing Outliers (IQR):\n", df_no_outlier_iqr)

# 3. Visualization
sns.boxplot(df['Salary'])
plt.show()
