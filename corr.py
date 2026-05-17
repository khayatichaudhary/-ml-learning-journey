import pandas as pd
df=pd.read_csv("train_and_test2.csv")
print("relation between these:",df.corr())
# Total females
total_females = len(df[df["Sex"] == 1])

# Female survivors
female_survivors = len(df[(df["Sex"] == 1) & (df["2urvived"] == 1)])

# Probability
probability = female_survivors / total_females
print("Survival probability if female:", probability)

# Now for males
total_males = len(df[df["Sex"] == 0])
male_survivors = len(df[(df["Sex"] == 0) & (df["2urvived"] == 1)])
probability_male = male_survivors / total_males
print("Survival probability if male:", probability_male)