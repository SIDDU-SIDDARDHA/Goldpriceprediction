from flask import Flask, render_template, request
import pickle
import pandas as pd
import datetime

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get date from form
        date_str = request.form['date']
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")

        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        day_of_year = date_obj.timetuple().tm_yday

        # Create input for ML model
        input_data = pd.DataFrame([[year, month, day, day_of_year]],
                                  columns=['Year','Month','Day','DayOfYear'])

        # Predict 24K price per 10gm
        pred_24k_10gm = model.predict(input_data)[0]

        # Convert to other prices
        pred_24k_1gm = pred_24k_10gm / 10
        pred_916_1gm = pred_24k_1gm * 0.916
        pred_916_10gm = pred_24k_10gm * 0.916

        return render_template("index.html",
            prediction_text=f"""
            Predicted Gold Prices for {date_str}

            24K Gold:
            1 Gram  : ₹{pred_24k_1gm:.2f}
            10 Gram : ₹{pred_24k_10gm:.2f}

            916 Gold:
            1 Gram  : ₹{pred_916_1gm:.2f}
            10 Gram : ₹{pred_916_10gm:.2f}
            """
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
