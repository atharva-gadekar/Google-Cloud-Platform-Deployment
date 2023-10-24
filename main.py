import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from pyngrok import ngrok

app = Flask(__name__)
model = pickle.load(open('randomForestRegressor.pkl', 'rb'))

# Set the desired port number for the Flask app
port_no = 5000

# Input the ngrok authentication token
ngrok_auth_token = input("Enter your ngrok authentication token: ")
ngrok.set_auth_token(ngrok_auth_token)

# Connect to ngrok
public_url = ngrok.connect(port_no).public_url

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction[0])

    return render_template('home.html', prediction_text="AQI for Jaipur {}".format(prediction[0]))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    print(f"To access the Global link, please click {public_url}")
    app.run(port=port_no, debug=True)
