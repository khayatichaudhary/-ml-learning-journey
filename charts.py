import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("train_and_test2.csv")

# Chart 1 - How many survived vs died
df["2urvived"].value_counts().plot(kind="bar", 
                                   color=["red", "green"])
plt.title("Survived vs Died on Titanic")
plt.xlabel("0 = Died, 1 = Survived")
plt.ylabel("Number of People")
plt.show()

# Chart 2 - Male vs Female
df["Sex"].value_counts().plot(kind="bar",
                               color=["blue", "pink"])
plt.title("Male vs Female on Titanic")
plt.xlabel("0 = Male, 1 = Female")
plt.ylabel("Number of People")
plt.show()

# Chart 3 - Age of all passengers
df["Age"].plot(kind="hist", 
               color="purple",
               bins=20)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.show()