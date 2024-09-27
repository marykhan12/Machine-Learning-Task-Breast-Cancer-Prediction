from flask import Flask, request, render_template
import numpy as np
import joblib
import sys
sys.path.append('D:\Machine Learning\Breast cancer prediction\templates\breast_cancer.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model (replace with the correct path to your model file)
model = joblib.load(r'D:\Machine Learning\Breast cancer prediction\templates\breast_cancer.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = [float(x) for x in request.form.values()]
    
    # Convert input data to numpy array and reshape
    input_data_np = np.array(input_data).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data_np)
    
    # Interpret the result
    result = 'Malignant' if prediction[0] == 0 else 'Benign'
    
    # Return the result to the UI
    return render_template('result.html', prediction_text=f'The Breast Cancer is {result}')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
