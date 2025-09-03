from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load saved model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Iris Prediction API is running 🚀"

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting JSON input
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
