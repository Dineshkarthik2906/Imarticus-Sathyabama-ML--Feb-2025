# Imarticus-Sathyabama-ML--Feb-2025
---
# AI-Powered Investment Advisor
 An AI-driven tool to predict portfolio allocations using XGBoost. It uses real-world market data and synthetic user profiles for training.

## Features
#Fetches market data (S&P 500, Bonds, Gold, Bitcoin) via Yahoo Finance.

#Generates synthetic user profiles with risk-based allocations.

#Trains an XGBoost model with GridSearchCV for hyperparameter tuning.

#Predicts portfolio allocations (Stocks, Bonds, Gold, Crypto).

---
## Installation and Running the Project

1.Train the Model:

python train_model.py

2.Run the Flask App:

python app.py

---
## Metrics

### Training Set:

MAE: 3.64

MSE: 20.87

R²: 0.57

### Test Set:

MAE: 3.71

MSE: 21.80

R²: 0.55

---
## File Structure
Investment-Advisor/ ( MAIN FOLDER)

├── app.py                  # Flask app

├── train_model.py          # Model training script

├── xgboost_model.pkl       # Trained model

└── templates/              ( SUB FOLDER INSIDE MAIN FOLDER)

   ├── index.html          # Input form
   
---
## Dependencies

### Python 3.8+

### Libraries:

numpy

pandas

yfinance

xgboost

scikit-learn

joblib

Flask



