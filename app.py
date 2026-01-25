import streamlit as st
import numpy as np
import pandas as pd

from embedded_data import load_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge

# --------------------------------------------------
st.set_page_config(
    page_title="Sediment Transport Predictor",
    page_icon="⛰️",
    layout="centered"
)

# --------------------------------------------------
st.markdown("""
<h2 style="text-align:center;">⛰️ GUI Tool for Sediment Transport Prediction In Steep Mountain Channels </h2>
<p style="text-align:center;color:gray;">

</p>
<hr>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load embedded confidential data
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
st.subheader("⚙️ Input Hydraulic Parameters")

col1, col2 = st.columns(2)

with col1:
    So = st.number_input("📐 Bed slope (So)", value=0.03, format="%.4f")
    Q = st.number_input("💧 Discharge Q (m³/s)", value=0.15, format="%.3f")
    U = st.number_input("➡️ Mean velocity U (m/s)", value=1.40, format="%.2f")
    H = st.number_input("📏 Flow depth H (m)", value=0.23, format="%.2f")

with col2:
    Re = st.number_input("🔁 Reynolds number", value=3.2e5, format="%.1e")
    theta = st.number_input("⚖️ Shields parameter (θ)", value=0.001, format="%.5f")
    lambda_D = st.number_input("🪨 Relative boulder spacing (λ/D)", value=2.0, format="%.2f")

X_new = np.array([[So, Q, U, H, Re, theta, lambda_D]])

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# Predict button
if st.button("🔮 Predict Sediment Transport", use_container_width=True):

    st.markdown("### 📊 Predicted Dimensionless Bedload Transport Rate (Φ)")

    colA, colB = st.columns(2)

    with colA:
        phi_bma = models["BMA"].predict(X_new)[0]
        phi_gbr = models["GBR"].predict(X_new)[0]

        st.success(f"🔵 **BMA**: {phi_bma:.2e}")
        st.success(f"🟢 **GBR**: {phi_gbr:.2e}")

    with colB:
        phi_knn = models["KNN"].predict(X_new)[0]
        phi_gpr, std = models["GPR"].predict(X_new, return_std=True)

        st.success(f"🟠 **KNN**: {phi_knn:.2e}")
        st.success(f"🟣 **GPR**: {phi_gpr[0]:.2e}")

        st.caption(f"📈 GPR uncertainty (±1σ): {std[0]:.2e}")

# --------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center; font-size:13px; color:gray;">
        <b>Developed by Ajaz Ahmad Mir</b><br>
        Research Scholar, Department of Civil Engineering<br>
        Dr. B. R. Ambedkar National Institute of Technology, Jalandhar, India
    </div>
    """,
    unsafe_allow_html=True
)

