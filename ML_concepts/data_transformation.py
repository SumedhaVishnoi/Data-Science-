import pandas as pd

df = pd.DataFrame({'Age': ['25', '30', '35']})
df['Age'] = df['Age'].astype(int)   # string to integer
print(df.dtypes)

df = pd.DataFrame({'Marks': [85.5, 90.0, 76.2]})
df['Marks'] = df['Marks'].astype(int)   # float to int
print(df)

df = pd.DataFrame({'Join_Date': ['2023-01-15', '2023-05-20']})
df['Join_Date'] = pd.to_datetime(df['Join_Date'])
print(df.dtypes)

df = pd.DataFrame({'Age': [12, 20, 35, 70]})
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100],
                         labels=['Child', 'Young Adult', 'Adult', 'Senior'])
print(df)


df = pd.DataFrame({'City': ['Delhi', 'Mumbai', 'Delhi', 'Bangalore']})
df['City'] = df['City'].astype('category')
print(df.dtypes)


# Check datatypes
print(df.dtypes)

# Convert column datatype
df['col'] = df['col'].astype(float)

# Convert multiple columns
df = df.astype({'Age': int, 'Salary': float})
