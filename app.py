import os
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained CNN model from the H5 file
model_path = os.path.join(os.getcwd(), 'earthquake.h5')
model = load_model(model_path)

# Function to preprocess input and make predictions


def predict_magnitude(input_data):
    # Preprocess input data (you may need to adapt this to your model's input requirements)
    input_data = np.array(input_data).reshape(1, -1)
    # Make predictions using the loaded model
    predictions = model.predict(input_data)
    # Assuming predictions is a 2D array with one row and multiple columns
    # You can take the mean of the predictions
    # print(predictions[0][0])
    prediction = np.mean(predictions)

    return predictions


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None

    if request.method == 'POST':
        # Get user input from the form
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        timestamp = float(request.form['timestamp'])

        # Prepare input data for prediction
        input_data = [latitude, longitude, timestamp]

        # Get the model prediction
        prediction_result = predict_magnitude(input_data)

    print(prediction_result)
    return render_template('index.html', prediction_result=prediction_result)


if __name__ == '__main__':
    app.run(debug=True)
