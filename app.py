from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open("diabetes_model.pkl", "rb"))

@app.route("/")
def homePage():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get features from the form
    float_features = [float(x) for x in request.form.values()] 
    features = [np.array(float_features)]
    
    # Predict the outcome using the loaded model
    prediction = model.predict(features)
    
    # Display the prediction result
    return render_template("index.html", prediction_details=f"The model predicts that the patient has {'diabetes' if prediction == 1 else 'no diabetes'}.")
    
if __name__ == "__main__":
    app.run(debug=True)
