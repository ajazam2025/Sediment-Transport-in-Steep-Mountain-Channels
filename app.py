import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from embedded_data import load_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sediment Transport Dashboard",
    page_icon="🌊",
    layout="wide"
)

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>🌊 Sediment Transport Prediction Dashboard</h1>
    <h4 style='text-align: center; color: gray;'>
    Steep Mountain Channels | Experimental–ML Framework
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- LOAD DATA ----------------
df = load_data()

X = df[["So", "Q", "U", "H", "Re", "theta", "lambda_D"]]
y = df["Phi"]

# ---------------- TRAIN MODELS ----------------
@st.cache_resource
def train_models():
    return {
        "BMA": BayesianRidge().fit(X, y),
        "GBR": GradientBoostingRegressor().fit(X, y),
        "GPR": GaussianProcessRegressor().fit(X, y),
        "KNN": KNeighborsRegressor(n_neighbors=3).fit(X, y),
    }

models = train_models()

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Hydraulic Inputs")

So = st.sidebar.slider("📐 Bed Slope (So)", 0.001, 0.05, 0.03)
Q = st.sidebar.slider("💧 Discharge Q (m³/s)", 0.1, 0.3, 0.15)
U = st.sidebar.slider("➡️ Mean Velocity U (m/s)", 0.5, 3.0, 1.4)
H = st.sidebar.slider("📏 Flow Depth H (m)", 0.2, 0.3, 0.23)

Re = st.sidebar.number_input("🔁 Reynolds Number", value=3.0e5, step=1e4)
theta = st.sidebar.slider("⚖️ Shields Parameter (θ)", 1e-5, 0.005, 0.001)
lambda_D = st.sidebar.slider("🪨 Relative Boulder Spacing (λ/D)", 1.0, 2.5, 2.0)

model_choice = st.sidebar.radio(
    "🧠 Select Model",
    ["BMA", "GBR", "GPR", "KNN"]
)

X_new = np.array([[So, Q, U, H, Re, theta, lambda_D]])

# ================= MAIN DASHBOARD =================
st.subheader("📊 Prediction Results")

col1, col2, col3, col4 = st.columns(4)

predictions = {}
for name, model in models.items():
    if name == "GPR":
        pred, std = model.predict(X_new, return_std=True)
        predictions[name] = (pred[0], std[0])
    else:
        predictions[name] = (model.predict(X_new)[0], None)

# ---- METRIC CARDS ----
col1.metric("🔵 BMA Φ", f"{predictions['BMA'][0]:.2e}")
col2.metric("🟢 GBR Φ", f"{predictions['GBR'][0]:.2e}")
col3.metric("🟣 GPR Φ", f"{predictions['GPR'][0]:.2e}")
col4.metric("🟠 KNN Φ", f"{predictions['KNN'][0]:.2e}")

if predictions["GPR"][1] is not None:
    st.info(f"📈 GPR Uncertainty (±1σ): {predictions['GPR'][1]:.2e}")

st.markdown("---")

# ================= MODEL COMPARISON PLOT =================
st.subheader("📈 Model Comparison")

model_names = list(predictions.keys())
phi_values = [predictions[m][0] for m in model_names]

fig = go.Figure(
    data=[
        go.Bar(
            x=model_names,
            y=phi_values,
            text=[f"{v:.2e}" for v in phi_values],
            textposition="outside",
            marker=dict(line=dict(width=1))
        )
    ]
)

fig.update_layout(
    xaxis_title="Machine Learning Models",
    yaxis_title="Predicted Sediment Transport (Φ)",
    template="simple_white",
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# ================= TIME EVOLUTION =================
st.subheader("⏱️ Time Evolution of Sediment Transport")

time_series = df.groupby("time")["Phi"].mean().reset_index()

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=time_series["time"],
        y=time_series["Phi"],
        mode="lines+markers",
        name="Mean Φ",
        line=dict(width=3)
    )
)

fig2.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Mean Sediment Transport (Φ)",
    template="simple_white",
    height=400
)

st.plotly_chart(fig2, use_container_width=True)

# ================= FOOTER =================
st.markdown("---")
st.caption(
    "🧪 Models trained at runtime using embedded experimental data. "
    "Designed for research, reproducibility, and confidentiality."
)
