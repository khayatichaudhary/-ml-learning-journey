import pandas as pd

df = pd.read_csv("train_and_test2.csv")

# See first 5 rows
print(df.head())

# How many rows and columns
print(df.shape)

# Basic statistics
print(df.describe())

# Check missing values
print(df.isnull().sum())

# How many people survived?
print(df["2urvived"].value_counts())

# Average age of survivors
survivors = df[df["2urvived"] == 1]
print("Average age of survivors:", survivors["Age"].mean())

# How many male vs female
print(df["Sex"].value_counts())