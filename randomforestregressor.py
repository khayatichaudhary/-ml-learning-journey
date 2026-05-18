import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#load data
df=pd.read_csv("train_and_test2.csv")
#fix missing value
df["Age"] = df["Age"].fillna(df["Age"].mean())
#chcek correlation of fare with features
print(df.corr()["Fare"].sort_values(ascending=False))
X=df[["Pclass","Age","Sex","Parch"]]
y=df["Fare"]
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#create linear regression model
# model = LinearRegression()
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
#fit model
model.fit(X_train, y_train)
#predict
predictions = model.predict(X_test)
#check score
print("Score:", model.score(X_test, y_test))

