import pandas as pd
import numpy as np
df=pd.read_csv("train_and_test2.csv")
print("mean age:",df["Age"].mean())
print("median age:",df["Age"].median())
print("mode age:",df["Age"].mode()[0])
print("standard deviation:",df["Age"].std())
print("min age:", df["Age"].min())
print("max age:", df["Age"].max())
