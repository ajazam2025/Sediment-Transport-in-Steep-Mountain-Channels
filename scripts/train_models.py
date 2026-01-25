import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge

df = pd.read_csv("data/sediment_transport_full.csv")

X = df[["So (%)","Q (m3/s)","U (m/s)","H (m)","Re","θ","H/D","λ/D"]]
y = df["Φ"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

bma = BayesianRidge().fit(X_train, y_train)
gbr = GradientBoostingRegressor().fit(X_train, y_train)
gpr = GaussianProcessRegressor().fit(X_train, y_train)
knn = KNeighborsRegressor().fit(X_train, y_train)

joblib.dump(bma, "models/bma_model.pkl")
joblib.dump(gbr, "models/gbr_model.pkl")
joblib.dump(gpr, "models/gpr_model.pkl")
joblib.dump(knn, "models/knn_model.pkl")
