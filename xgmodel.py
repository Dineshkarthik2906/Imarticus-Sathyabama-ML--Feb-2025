import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Fetch market data
def fetch_market_data():
    tickers = ["^GSPC", "BND", "GLD", "BTC-USD"]  # S&P 500, Bonds, Gold, Bitcoin
    data = yf.download(tickers, start="2020-01-01", end="2024-03-01")["Close"]
    returns = data.pct_change().dropna()  # Calculate daily returns
    trends = returns.rolling(window=7).mean().dropna()  # 7-day rolling average trends
    return trends

# Generate synthetic user profiles with real market trends
def generate_user_profiles(n=20000):  # Increased dataset size to 20,000
    np.random.seed(42)

    # Fetch market data
    market_data = fetch_market_data()

    # Generate synthetic user data
    age = np.random.randint(18, 65, n)
    income = np.random.randint(20000, 150000, n)
    risk_tolerance = np.random.choice(["Low", "Medium", "High"], n, p=[0.3, 0.5, 0.2])
    literacy_rate = np.random.uniform(0.5, 1.0, n)  # Literacy rate between 50% and 100%
    investment_goal = np.random.choice(["Retirement", "Wealth Accumulation", "Education", "Home Purchase"], n)

    # Sample market trends from the fetched data
    sample_indices = np.random.choice(len(market_data), n)
    stock_trend = market_data.iloc[sample_indices, 0].values  # S&P 500 trend
    bond_trend = market_data.iloc[sample_indices, 1].values  # Bonds trend
    gold_trend = market_data.iloc[sample_indices, 2].values  # Gold trend
    crypto_trend = market_data.iloc[sample_indices, 3].values  # Bitcoin trend

    # Portfolio allocation based on risk tolerance with added noise
    portfolio_allocation = []

    for risk in risk_tolerance:
        if risk == "Low":
            base_allocation = [10, 70, 15, 5]  # More bonds, less crypto
        elif risk == "Medium":
            base_allocation = [40, 40, 15, 5]  # Balanced
        else:  # High risk
            base_allocation = [70, 10, 10, 10]  # More stocks & crypto

        # Add noise to the allocation
        noise = np.random.normal(0, 5, 4)  # Adding noise with mean 0 and std dev 5
        allocation = np.clip(base_allocation + noise, 0, 100)  # Ensure allocations are between 0% and 100%
        allocation = allocation / np.sum(allocation) * 100  # Normalize to sum to 100%
        portfolio_allocation.append(allocation)

    portfolio_allocation = np.array(portfolio_allocation)  # Convert to NumPy array

    # Create DataFrame
    df = pd.DataFrame({
        "Age": age,
        "Income": income,
        "Risk_Tolerance": risk_tolerance,
        "Literacy_Rate": literacy_rate,
        "Investment_Goal": investment_goal,
        "Stock_Trend": stock_trend,
        "Bond_Trend": bond_trend,
        "Gold_Trend": gold_trend,
        "Crypto_Trend": crypto_trend,
        "Stocks_%": portfolio_allocation[:, 0],
        "Bonds_%": portfolio_allocation[:, 1],
        "Gold_%": portfolio_allocation[:, 2],
        "Crypto_%": portfolio_allocation[:, 3],
    })

    # Encode categorical features
    df["Risk_Tolerance"] = df["Risk_Tolerance"].map({"Low": 0, "Medium": 1, "High": 2})
    df["Investment_Goal"] = df["Investment_Goal"].map({
        "Retirement": 0,
        "Wealth Accumulation": 1,
        "Education": 2,
        "Home Purchase": 3
    })

    return df

# Train and evaluate the XGBoost model
def train_xgboost_model():
    # Generate synthetic data
    user_data = generate_user_profiles(n=20000)  # Increased dataset size to 20,000

    # Features and target
    X = user_data[["Age", "Income", "Risk_Tolerance", "Literacy_Rate", "Investment_Goal", "Stock_Trend", "Bond_Trend", "Gold_Trend", "Crypto_Trend"]]
    Y = user_data[["Stocks_%", "Bonds_%", "Gold_%", "Crypto_%"]]

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize XGBoost model
    xgb_model = XGBRegressor(random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    # Save the trained model
    joblib.dump(best_model, "xgboost_model.pkl")
    print("Model trained and saved.")

    # Evaluate the model
    Y_pred = best_model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)
    print(f"Model Evaluation:")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")

    return best_model
