import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from embedded_data import load_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge

# --------------------------------------------------
st.set_page_config(
    page_title="Sediment Transport Dashboard",
    page_icon="🌊",
    layout="wide"
)

# --------------------------------------------------
st.markdown("""
<h1 style='text-align:center;'>🌊 Sediment Transport Prediction Dashboard</h1>
<h4 style='text-align:center;color:gray;'>
Steep Mountain Channels | Experimental + Machine Learning
</h4>
""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
df = load_data()

X = df[["So","Q","U","H","Re","theta","lambda_D"]]
y = df["Phi"]

@st.cache_resource
def train_models():
    return {
        "BMA": BayesianRidge().fit(X, y),
        "GBR": GradientBoostingRegressor().fit(X, y),
        "GPR": GaussianProcessRegressor().fit(X, y),
        "KNN": KNeighborsRegressor(n_neighbors=3).fit(X, y)
    }

models = train_models()

# --------------------------------------------------
st.sidebar.header("⚙️ Hydraulic Inputs")

So = st.sidebar.slider("📐 Bed slope (So)", 0.001, 0.05, 0.03)
Q = st.sidebar.slider("💧 Discharge Q (m³/s)", 0.1, 0.3, 0.15)
U = st.sidebar.slider("➡️ Velocity U (m/s)", 0.5, 3.0, 1.4)
H = st.sidebar.slider("📏 Flow depth H (m)", 0.2, 0.3, 0.23)
Re = st.sidebar.number_input("🔁 Reynolds number", value=3.2e5)
theta = st.sidebar.slider("⚖️ Shields parameter θ", 1e-5, 0.005, 0.001)
lambda_D = st.sidebar.slider("🪨 Relative spacing λ/D", 1.0, 2.5, 2.0)

X_new = np.array([[So,Q,U,H,Re,theta,lambda_D]])

# --------------------------------------------------
st.subheader("📊 Model Predictions")

cols = st.columns(4)
preds = {}

for name, model in models.items():
    if name == "GPR":
        val, std = model.predict(X_new, return_std=True)
        preds[name] = (val[0], std[0])
    else:
        preds[name] = (model.predict(X_new)[0], None)

cols[0].metric("🔵 BMA Φ", f"{preds['BMA'][0]:.2e}")
cols[1].metric("🟢 GBR Φ", f"{preds['GBR'][0]:.2e}")
cols[2].metric("🟣 GPR Φ", f"{preds['GPR'][0]:.2e}")
cols[3].metric("🟠 KNN Φ", f"{preds['KNN'][0]:.2e}")

if preds["GPR"][1] is not None:
    st.info(f"📈 GPR Uncertainty (±1σ): {preds['GPR'][1]:.2e}")

st.markdown("---")

# --------------------------------------------------
st.subheader("📈 Model Comparison")

fig = go.Figure(
    go.Bar(
        x=list(preds.keys()),
        y=[preds[m][0] for m in preds],
        text=[f"{preds[m][0]:.2e}" for m in preds],
        textposition="outside"
    )
)

fig.update_layout(
    template="simple_white",
    xaxis_title="Machine Learning Models",
    yaxis_title="Predicted Φ"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
st.subheader("⏱️ Time Evolution of Sediment Transport")

time_series = df.groupby("time")["Phi"].mean().reset_index()

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=time_series["time"],
        y=time_series["Phi"],
        mode="lines+markers",
        line=dict(width=3)
    )
)

fig2.update_layout(
    template="simple_white",
    xaxis_title="Time (s)",
    yaxis_title="Mean Φ"
)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
st.markdown("---")
st.caption(
    "🧪 Models trained dynamically using embedded confidential data. "
    "Designed for reproducible research and reviewer-safe deployment."
)
