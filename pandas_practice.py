import pandas as pd

# Creating a simple table of data
data = {
    "Name": ["Khayati", "Rahul", "Priya", "Arjun"],
    "Age": [20, 21, 19, 22],
    "College": ["MNNIT", "IIT", "NIT", "BITS"],
    "Goal": ["AI Engineer", "Data Scientist", 
             "ML Engineer", "AI Researcher"]
}


df = pd.DataFrame(data)

# Printing the table
print(df)
print(df["Name"])

print(df.head(2))

print(df.shape)

print(df.info())

print(df[df["Name"] == "Khayati"])

df["City"] = ["Agra", "Delhi", "Mumbai", "Pune"]
print(df)