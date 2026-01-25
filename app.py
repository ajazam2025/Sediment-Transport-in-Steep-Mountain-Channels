import streamlit as st
import numpy as np
import joblib
from embedded_data import load_data

st.set_page_config(layout="wide")
st.title("Sediment Transport Prediction in Steep Channels")

df = load_data()

st.markdown("### Input Hydraulic Parameters")

col1, col2, col3 = st.columns(3)

So = col1.number_input("Bed Slope (So)", 0.001, 0.05, 0.03)
Q = col2.number_input("Discharge Q (m³/s)", 0.1, 0.3, 0.15)
U = col3.number_input("Mean Velocity U (m/s)", 0.5, 3.0, 1.4)

H = col1.number_input("Flow Depth H (m)", 0.2, 0.3, 0.23)
Re = col2.number_input("Reynolds Number", 2e5, 7e5, 3e5)
theta = col3.number_input("Shields Parameter θ", 1e-5, 0.005, 0.001)

lambda_D = st.slider("Relative Boulder Spacing (λ/D)", 1.0, 2.5, 2.5)

X = np.array([[So, Q, U, H, Re, theta, lambda_D]])

model_choice = st.selectbox(
    "Select Model",
    ["BMA", "GBR", "GPR", "KNN"]
)

if st.button("Predict Sediment Transport"):
    model = joblib.load(f"{model_choice}_model.pkl")

    if model_choice == "GPR":
        y, std = model.predict(X, return_std=True)
        st.success(f"Predicted Φ: {y[0]:.4e}")
        st.info(f"Uncertainty ±1σ: {std[0]:.4e}")
    else:
        y = model.predict(X)[0]
        st.success(f"Predicted Φ: {y:.4e}")
