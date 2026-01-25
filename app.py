import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="wide")
st.title("Sediment Transport Prediction in Steep Channels")

# Load data
df = pd.read_csv("data/sediment_transport_full.csv")

# Load models
bma = joblib.load("models/bma_model.pkl")
gbr = joblib.load("models/gbr_model.pkl")
gpr = joblib.load("models/gpr_model.pkl")
knn = joblib.load("models/knn_model.pkl")

# Time selection
time_selected = st.selectbox("Select Time (s)", sorted(df["time (s)"].unique()))
df_t = df[df["time (s)"] == time_selected]

st.markdown("### Input Parameters")

cols = st.columns(4)
So = cols[0].number_input("So (%)", value=0.03)
Q = cols[1].number_input("Q (m³/s)", value=0.15)
U = cols[2].number_input("U (m/s)", value=1.4)
H = cols[3].number_input("H (m)", value=0.23)

cols = st.columns(4)
Re = cols[0].number_input("Re", value=3.2e5)
theta = cols[1].number_input("θ", value=0.001)
HD = cols[2].number_input("H/D", value=1.65)
lambdaD = cols[3].number_input("λ/D", value=2.5)

X = np.array([[So, Q, U, H, Re, theta, HD, lambdaD]])

model = st.selectbox(
    "Select Model",
    ["Bayesian Model Averaging (BMA)",
     "Gradient Boosting Regression (GBR)",
     "Gaussian Process Regression (GPR)",
     "k-Nearest Neighbours (KNN)"]
)

if st.button("Predict Sediment Transport"):
    if model == "Bayesian Model Averaging (BMA)":
        y = bma.predict(X)[0]
        st.success(f"BMA Prediction (Φ): {y:.4e}")

    elif model == "Gradient Boosting Regression (GBR)":
        y = gbr.predict(X)[0]
        st.success(f"GBR Prediction (Φ): {y:.4e}")

    elif model == "Gaussian Process Regression (GPR)":
        y, std = gpr.predict(X, return_std=True)
        st.success(f"GPR Prediction (Φ): {y[0]:.4e}")
        st.info(f"Uncertainty ±1σ: {std[0]:.4e}")

    elif model == "k-Nearest Neighbours (KNN)":
        y = knn.predict(X)[0]
        st.success(f"KNN Prediction (Φ): {y:.4e}")
