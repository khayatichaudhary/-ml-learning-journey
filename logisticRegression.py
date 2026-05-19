import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#load data
df=pd.read_csv("train_and_test2.csv")
X=df[["Sex","Fare","Pclass"]]
y=df["2urvived"]
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
#fit model
model.fit(X_train, y_train)
#predict
predictions = model.predict(X_test)
# check score
print("Score:", model.score(X_test, y_test))
