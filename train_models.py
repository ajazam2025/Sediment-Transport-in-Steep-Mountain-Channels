from embedded_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge
import joblib

df = load_data()

X = df[["So", "Q", "U", "H", "Re", "theta", "lambda_D"]]
y = df["Phi"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "BMA": BayesianRidge(),
    "GBR": GradientBoostingRegressor(),
    "GPR": GaussianProcessRegressor(),
    "KNN": KNeighborsRegressor(n_neighbors=3)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}_model.pkl")

print("Models trained and saved locally.")
