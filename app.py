import streamlit as st
import numpy as np

from embedded_data import load_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

# --------------------------------------------------
st.set_page_config(
    page_title="Sediment Transport Predictor",
    page_icon="⛰️",
    layout="centered"
)

st.markdown("""
<h2 style="text-align:center;">⛰️ GUI Tool for Sediment Transport Prediction</h2>
<p style="text-align:center;color:gray;">
Steep Mountain Channels | Experimental–ML Framework
</p>
<hr>
""", unsafe_allow_html=True)

# --------------------------------------------------
df = load_data()

X = df[["So","Q","U","H","Re","theta","lambda_D"]].values
y = df["Phi"].values
y_log = np.log10(y + 1e-12)   # IMPORTANT

# --------------------------------------------------
@st.cache_resource
def train_models():

    models = {
        "BMA": BayesianRidge().fit(X, y),
        "GBR": GradientBoostingRegressor(random_state=42).fit(X, y),
        "KNN": KNeighborsRegressor(n_neighbors=3).fit(X, y),
    }

    # GPR ONLY IN LOG SPACE (diagnostic)
    gpr_kernel = (
        ConstantKernel(1.0) *
        Matern(length_scale=np.ones(X.shape[1]), nu=1.5) +
        WhiteKernel(noise_level=1e-2)
    )

    gpr = GaussianProcessRegressor(
        kernel=gpr_kernel,
        alpha=1e-2,
        normalize_y=False
    )

    gpr.fit(X, y_log)
    models["GPR_LOG"] = gpr

    return models

models = train_models()

# --------------------------------------------------
st.subheader("⚙️ Input Parameters")

col1, col2 = st.columns(2)

with col1:
    So = st.number_input("Bed slope (So)", value=0.03)
    Q = st.number_input("Discharge Q (m³/s)", value=0.15)
    U = st.number_input("Velocity U (m/s)", value=1.40)
    H = st.number_input("Flow depth H (m)", value=0.23)

with col2:
    Re = st.number_input("Reynolds number", value=3.2e5, format="%.1e")
    theta = st.number_input("Shields parameter θ", value=0.001)
    lambda_D = st.number_input("Relative spacing λ/D", value=2.0)

X_new = np.array([[So, Q, U, H, Re, theta, lambda_D]])

# --------------------------------------------------
if st.button("🔮 Predict Sediment Transport", use_container_width=True):

    st.markdown("### 📊 Predicted Bedload Transport")

    phi_bma = models["BMA"].predict(X_new)[0]
    phi_gbr = models["GBR"].predict(X_new)[0]
    phi_knn = models["KNN"].predict(X_new)[0]
    phi_gpr_log = models["GPR_LOG"].predict(X_new)[0]

    colA, colB = st.columns(2)

    with colA:
        st.success(f"🔵 **BMA Φ** : {phi_bma:.2e}")
        st.success(f"🟢 **GBR Φ** : {phi_gbr:.2e}")

    with colB:
        st.success(f"🟠 **KNN Φ** : {phi_knn:.2e}")
        st.info(f"🟣 **GPR (log₁₀ Φ)** : {phi_gpr_log:.2f}")

        st.caption(
            "GPR is shown in log-space due to numerical instability "
            "for highly heterogeneous sediment transport data."
        )

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
