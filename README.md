# Sediment Transport Prediction in Steep Channels

This repository provides a Streamlit-based graphical user interface
for predicting sediment transport using machine learning models.

## Models
- Bayesian Model Averaging (BMA)
- Gradient Boosting Regression (GBR)
- Gaussian Process Regression (GPR)
- k-Nearest Neighbours (KNN)

## Data Policy
The experimental dataset is embedded within the source code to
preserve confidentiality. No external data files are required.

## Run the App
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
