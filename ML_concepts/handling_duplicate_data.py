'''Duplicate rows = same data repeated.

This can bias the model by giving more weight to certain observations.

Especially problematic in classification/regression datasets.'''

import pandas as pd

# Sample dataset
data = {
    'ID': [1, 2, 2, 3, 4],
    'Name': ['John', 'Alice', 'Alice', 'Bob', 'Eve'],
    'Age': [25, 30, 30, 35, 40],
    'Salary': [50000, 60000, 60000, 70000, 80000]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# Check duplicates
print("\nDuplicate Rows:\n", df[df.duplicated()])

# Remove duplicates (keep first occurrence)
df_no_dup = df.drop_duplicates()
print("\nAfter Removing Duplicates:\n", df_no_dup)

# Keep last occurrence instead
df_last = df.drop_duplicates(keep='last')
print("\nKeep Last Occurrence:\n", df_last)

# Remove duplicates based on specific columns (only Name & Age)
df_subset = df.drop_duplicates(subset=['Name', 'Age'])
print("\nRemove Duplicates by Subset:\n", df_subset)
