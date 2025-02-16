from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Load Trained Model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(request.form[key]) for key in request.form.keys()]
        features = np.array(features).reshape(1, -1)
        
        # Predict Genre
        genre = model.predict(features)[0]
        
        return render_template('result.html', prediction=genre)

if __name__ == '__main__':
    app.run(debug=True)
