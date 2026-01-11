import streamlit as st
import numpy as np
import joblib

# -----------------------------------
# Load trained model
# -----------------------------------
model = joblib.load("gaussian_nb_titanic_model.pkl")

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction (GaussianNB)")
st.write("Enter passenger details to predict survival.")

# -----------------------------------
# User Inputs (MATCH TRAINING FEATURES)
# -----------------------------------
p_class = st.selectbox(
    "Passenger Class",
    options=[1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    options=["Female", "Male"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.0
)

# Convert sex to encoded value (same as training)
sex_male = 1 if sex == "Male" else 0

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict Survival"):

    # EXACT feature order used during training
    input_data = np.array([[
        p_class,
        age,
        fare,
        sex_male
    ]])

    # Safety check (prevents feature mismatch forever)
    if input_data.shape[1] != model.n_features_in_:
        st.error(
            f"Model expects {model.n_features_in_} features, "
            f"but received {input_data.shape[1]}"
        )
    else:
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("🟢 Passenger is likely to SURVIVE")
        else:
            st.error("🔴 Passenger is NOT likely to survive")
