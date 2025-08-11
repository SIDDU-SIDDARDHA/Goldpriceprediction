from flask import Flask, render_template, request
import pickle
import pandas as pd
import datetime

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs
        date_str = request.form['date']
        spx = float(request.form['spx'])
        uso = float(request.form['uso'])
        slv = float(request.form['slv'])
        eurusd = float(request.form['eurusd'])

        # Convert date to numeric features
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        day_of_year = date_obj.timetuple().tm_yday

        # Create dataframe for prediction
        input_data = pd.DataFrame([[year, month, day, day_of_year, spx, uso, slv, eurusd]],
                                  columns=['Year', 'Month', 'Day', 'DayOfYear', 'SPX', 'USO', 'SLV', 'EUR/USD'])

        # Predict
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f"Predicted Gold Price: {prediction:.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)

