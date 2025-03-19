import numpy as np
import joblib
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load("xgboost_model.pkl")

# Home route to display the form
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input from the form
        age = int(request.form["age"])
        income = int(request.form["income"])
        risk_tolerance = request.form["risk_tolerance"]
        literacy_rate = float(request.form["literacy_rate"])
        investment_goal = request.form["investment_goal"]
        stock_trend = float(request.form["stock_trend"])
        bond_trend = float(request.form["bond_trend"])
        gold_trend = float(request.form["gold_trend"])
        crypto_trend = float(request.form["crypto_trend"])

        # Map risk tolerance to numerical value
        risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
        risk_tolerance_num = risk_mapping[risk_tolerance]

        # Map investment goal to numerical value
        goal_mapping = {
            "Retirement": 0,
            "Wealth Accumulation": 1,
            "Education": 2,
            "Home Purchase": 3
        }
        investment_goal_num = goal_mapping[investment_goal]

        # Prepare input for the model
        user_input = np.array([[
            age, income, risk_tolerance_num, literacy_rate, investment_goal_num,
            stock_trend, bond_trend, gold_trend, crypto_trend
        ]])

        # Make prediction
        allocation = model.predict(user_input)[0]

        # Prepare the result to display
        result = {
            "Stocks": f"{allocation[0]:.2f}%",
            "Bonds": f"{allocation[1]:.2f}%",
            "Gold": f"{allocation[2]:.2f}%",
            "Crypto": f"{allocation[3]:.2f}%"
        }

        # Render the result in the template
        return render_template("index.html", result=result)

    # Render the form for GET requests
    return render_template("index.html", result=None)

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)