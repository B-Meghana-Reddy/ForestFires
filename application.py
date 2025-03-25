import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

# Initialize Flask application
application = Flask(__name__)
app = application

# Load Ridge Regression Model and Scaler
ridge_model = pickle.load(open('ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler.pkl', 'rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Get form data
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Transform input data
            new_data_scaled = standard_scaler.transform(
                [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            )

            # Predict using the model
            result = float(ridge_model.predict(new_data_scaled)[0])

            return render_template('home.html', result=result)

        except Exception as e:
            return jsonify({'error': str(e)})

    # If method is GET, return home page
    return render_template('home.html', result=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
