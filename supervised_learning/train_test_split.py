'''In supervised learning, we need to check how well our model generalizes to unseen data.

If we train & test on the same data → model will just memorize (overfitting).

So we split dataset into:

Training Set → used to train the model

Test Set → used to evaluate the model
'''

import pandas as pd
from sklearn.model_selection import train_test_split

# Sample dataset
df = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, 50, 55, 60],
    'Salary': [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000],
    'Purchased': [0, 0, 0, 1, 1, 1, 1, 1]   # Target variable
})

# Features (X) and Target (y)
X = df[['Age', 'Salary']]
y = df['Purchased']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Features:\n", X_train)
print("\nTesting Features:\n", X_test)
print("\nTraining Target:\n", y_train)
print("\nTesting Target:\n", y_test)
