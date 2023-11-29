# Import the Flask class from the flask module
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)


# Load the pre-trained model
model_path = 'samskrta.h5'
model = load_model(model_path)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_input', methods=['POST'])
def process_input():
    user_input = request.form['user_input']
    prediction = model.predict(padded_sequence)[0][0]

    return render_template('result.html', user_input=user_input, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
