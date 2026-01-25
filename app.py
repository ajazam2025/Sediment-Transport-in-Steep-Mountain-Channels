import streamlit as st
import numpy as np
from embedded_data import load_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge

st.set_page_config(layout="wide")
st.title("Sediment Transport Prediction in Steep Channels")

# Load embedded data
df = load_data()

# Feature matrix and target
X = df[["So", "Q", "U", "H", "Re", "theta", "lambda_D"]]
y = df["Phi"]

# Train models ON THE FLY (fast for small datasets)
@st.cache_resource
def train_models():
    models = {
        "BMA": BayesianRidge(),
        "GBR": GradientBoostingRegressor(),
        "GPR": GaussianProcessRegressor(),
        "KNN": KNeighborsRegressor(n_neighbors=3)
    }
    for m in models.values():
        m.fit(X, y)
    return models

models = train_models()

# GUI inputs
st.markdown("### Input Hydraulic Parameters")

col1, col2, col3 = st.columns(3)

So = col1.number_input("Bed Slope (So)", 0.001, 0.05, 0.03)
Q = col2.number_input("Discharge Q (m³/s)", 0.1, 0.3, 0.15)
U = col3.number_input("Mean Velocity U (m/s)", 0.5, 3.0, 1.4)

H = col1.number_input("Flow Depth H (m)", 0.2, 0.3, 0.23)
Re = col2.number_input("Reynolds Number", 2e5, 7e5, 3e5)
theta = col3.number_input("Shields Parameter θ", 1e-5, 0.005, 0.001)

lambda_D = st.slider("Relative Boulder Spacing (λ/D)", 1.0, 2.5, 2.5)

X_new = np.array([[So, Q, U, H, Re, theta, lambda_D]])

model_choice = st.selectbox(
    "Select Model",
    ["BMA", "GBR", "GPR", "KNN"]
)

if st.button("Predict Sediment Transport"):
    model = models[model_choice]

    if model_choice == "GPR":
        y_pred, std = model.predict(X_new, return_std=True)
        st.success(f"Predicted Φ: {y_pred[0]:.4e}")
        st.info(f"Uncertainty ±1σ: {std[0]:.4e}")
    else:
        y_pred = model.predict(X_new)[0]
        st.success(f"Predicted Φ: {y_pred:.4e}")
