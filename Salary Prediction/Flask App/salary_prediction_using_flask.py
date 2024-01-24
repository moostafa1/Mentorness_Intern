from flask import Flask, render_template, request
import numpy as np
# import joblib
import pickle

app = Flask(__name__)
model = pickle.load(open('random_forest_grid.pickle', 'rb'))
# model = joblib.load('gradient_boosting_model.joblib')


@app.route('/')
def index():
    return render_template('render.html', data=None)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        val1 = float(request.form['AGE'])
        val2 = float(request.form['PAST_EXP'])
        val3 = float(request.form['AGE_Category'])
        val4 = float(request.form['DESIGNATION_Category'])

        # Validation to check for valid input values
        if any(np.isnan([val1, val2, val3, val4])):
            raise ValueError("Invalid input. Please enter numeric values for all fields.")

        arr = np.array([val1, val2, val3, val4], dtype=float)
        # arr = arr.astype(np.float64)
        pred = model.predict([arr])

        return render_template('render.html', data=int(pred))

    except ValueError as e:
        # Handle invalid input gracefully
        return render_template('render.html', data=None, error_message=str(e))


if __name__ == '__main__':
    app.run(debug=True)