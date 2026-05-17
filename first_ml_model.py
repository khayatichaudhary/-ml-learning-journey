import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("train_and_test2.csv")

# Use ONLY useful columns
X = df[["Age", "Fare", "Sex", "Pclass"]]
y = df["2urvived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

# Random Forest model
model = RandomForestClassifier(n_estimators=100,
                               random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy * 100, "%")

import numpy as np

# New passenger details
# Age, Fare, Sex (1=female, 0=male), Pclass
new_passenger = pd.DataFrame({
    "Age": [8], "Fare": [7.0], 
"Sex": [1], "Pclass": [3]
})

# Predict!
result = model.predict(new_passenger)

if result[0] == 1:
    print("This passenger SURVIVES! ✅")
else:
    print("This passenger DIES! ☠️")