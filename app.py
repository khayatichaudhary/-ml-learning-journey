import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Title
st.title("🚢 Titanic Survival Predictor")
# Add a description
st.markdown("""
This app uses **Machine Learning** (Random Forest) 
to predict Titanic survival based on:
- 👤 Age
- 💰 Ticket Fare  
- 👫 Gender
- 🎫 Passenger Class
""")

# Add a divider
st.divider()
st.write("Enter passenger details and I'll predict if they survive!")

# Load and train model automatically
df = pd.read_csv("train_and_test2.csv")
X = df[["Age", "Fare", "Sex", "Pclass"]]
y = df["2urvived"]
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, 
                                    test_size=0.2, 
                                    random_state=42)
model = RandomForestClassifier(n_estimators=100, 
                               random_state=42)
model.fit(X_train, y_train)

# User inputs
st.subheader("Enter Passenger Details:")

age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare (Ticket Price)", 0, 500, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
pclass = st.selectbox("Passenger Class", [1, 2, 3])

# Convert sex to number
sex_num = 1 if sex == "Female" else 0

# Predict button
if st.button("Predict Survival! 🔮"):
    passenger = pd.DataFrame({
        "Age": [age],
        "Fare": [fare],
        "Sex": [sex_num],
        "Pclass": [pclass]
    })
    
    result = model.predict(passenger)
    
    # Probability
    probability = model.predict_proba(passenger)
    survival_chance = probability[0][1] * 100
    
    if result[0] == 1:
        st.success("✅ This passenger SURVIVES!")
    else:
        st.error(" This passenger does NOT survive!")
    
    # Show percentage
    st.write(f"Survival Chance: **{survival_chance:.1f}%**")
    
    # Progress bar
    st.progress(int(survival_chance))

        


