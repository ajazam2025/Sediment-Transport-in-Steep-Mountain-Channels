import streamlit as st
import numpy as np

from embedded_data import load_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------
st.set_page_config(
    page_title="Sediment Transport Predictor",
    page_icon="⛰️",
    layout="centered"
)

# --------------------------------------------------
# FORCE background + centered rectangular card
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background-color: #eef1f4;
    }

    /* Main centered card */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stVerticalBlock"]) {
        background-color: white;
        padding: 35px 45px;
        border-radius: 14px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.12);
        max-width: 900px;
        margin: 30px auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Everything below is inside ONE Streamlit container
with st.container():

    st.markdown("""
    <h2 style="text-align:center;">⛰️ GUI Tool for Sediment Transport Prediction</h2>
    <p style="text-align:center; color:gray;">
    Steep Mountain Channels | Experimental–Machine Learning Framework
    </p>
    <hr>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    df = load_data()

    X = df[["So","Q","U","H","Re","theta","lambda_D"]].values
    y = df["Phi"].values

    # --------------------------------------------------
    @st.cache_resource
    def train_models():

        gbr = GradientBoostingRegressor(random_state=42)
        bma = BayesianRidge()
        knn = KNeighborsRegressor(n_neighbors=3)

        gbr.fit(X, y)
        bma.fit(X, y)
        knn.fit(X, y)

        # GPR trained on GBR residuals (stable)
        y_gbr = gbr.predict(X)
        residuals = y - y_gbr

        kernel = (
            ConstantKernel(1.0) *
            RBF(length_scale=np.ones(X.shape[1])) +
            WhiteKernel(noise_level=1e-3)
        )

        gpr = Pipeline([
            ("scaler", StandardScaler()),
            ("gpr", GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=3
            ))
        ])

        gpr.fit(X, residuals)

        return {
            "BMA": bma,
            "GBR": gbr,
            "KNN": knn,
            "GPR": gpr
        }

    models = train_models()

    # --------------------------------------------------
    st.subheader("⚙️ Input Hydraulic Parameters")

    col1, col2 = st.columns(2)

    with col1:
        So = st.number_input("📐 Bed slope (So)", value=0.03)
        Q = st.number_input("💧 Discharge Q (m³/s)", value=0.15)
        U = st.number_input("➡️ Mean velocity U (m/s)", value=1.40)
        H = st.number_input("📏 Flow depth H (m)", value=0.23)

    with col2:
        Re = st.number_input("🔁 Reynolds number", value=3.2e5, format="%.1e")
        theta = st.number_input("⚖️ Shields parameter (θ)", value=0.001)
        lambda_D = st.number_input("🪨 Relative boulder spacing (λ/D)", value=2.0)

    X_new = np.array([[So, Q, U, H, Re, theta, lambda_D]])

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------
    if st.button("🔮 Predict Sediment Transport", type="primary", use_container_width=True):

        st.markdown("### 📊 Predicted Dimensionless Bedload Transport Rate (Φ)")

        phi_bma = models["BMA"].predict(X_new)[0]
        phi_gbr = models["GBR"].predict(X_new)[0]
        phi_knn = models["KNN"].predict(X_new)[0]

        gpr_residual = models["GPR"].predict(X_new)[0]
        phi_gpr = phi_gbr + gpr_residual

        colA, colB = st.columns(2)

        with colA:
            st.success(f"🔵 **BMA** : {phi_bma:.2e}")
            st.success(f"🟢 **GBR** : {phi_gbr:.2e}")

        with colB:
            st.success(f"🟠 **KNN** : {phi_knn:.2e}")
            st.success(f"🟣 **GPR** : {phi_gpr:.2e}")

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
