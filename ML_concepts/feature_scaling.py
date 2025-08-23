# Many ML algorithms are distance-based (KNN, SVM, Gradient Descent) → larger range features dominate the model.

# Types of Feature Scaling
# 1. Normalization (Min-Max Scaling)
# Scales values between 0 and 1.

# Useful when distribution is not Gaussian and you need bounded values.

# 2.Standardization (Z-Score Scaling)
# Transforms data to mean = 0, std = 1.

# Works well with Gaussian distributions.

# Useful for algorithms assuming normality (Linear Regression, Logistic Regression, PCA).

# 3. Robust Scaling
# Uses median & IQR → robust to outliers.

# MaxAbs Scaling

# Scales values to [-1, 1] by dividing each value by the maximum absolute value.

# Useful for sparse data (like text data after vectorization)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

# Sample dataset
df = pd.DataFrame({
    'Age': [15, 20, 25, 30, 35],
    'Salary': [1000, 5000, 10000, 20000, 50000]
})
print("Original Data:\n", df)

# 1. Normalization (Min-Max)
mm = MinMaxScaler()
df['Age_MinMax'] = mm.fit_transform(df[['Age']])
df['Salary_MinMax'] = mm.fit_transform(df[['Salary']])

# 2. Standardization (Z-score)
sc = StandardScaler()
df['Age_Std'] = sc.fit_transform(df[['Age']])
df['Salary_Std'] = sc.fit_transform(df[['Salary']])

# 3. Robust Scaling
rb = RobustScaler()
df['Age_Robust'] = rb.fit_transform(df[['Age']])
df['Salary_Robust'] = rb.fit_transform(df[['Salary']])

print("\nAfter Scaling:\n", df)


