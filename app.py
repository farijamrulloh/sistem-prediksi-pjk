from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Load x_train from pickle file
with open('x_train.pkl', 'rb') as file:
    x_train = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediksi')
def prediksi():
    return render_template('prediksi.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/obat')
def obat():
    return render_template('obat.html')


@app.route('/hasil', methods=['POST'])
def hasil():
    # age	sex	cp	trtbps	chol	fbs	restecg	thalachh	exng	oldpeak	slp	caa	thall
    # Get input data from form
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trtbps = float(request.form['trtbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalachh = float(request.form['thalachh'])
    exng = float(request.form['exng'])
    oldpeak = float(request.form['oldpeak'])
    slp = float(request.form['slp'])
    caa = float(request.form['caa'])
    thall = float(request.form['thall'])

    input_data = np.array([[age, sex, cp, trtbps, chol, fbs,
                          restecg, thalachh, exng, oldpeak, slp, caa, thall]])

    # Normalize input data
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)

    # Render the result template with the prediction
    return render_template(
        'hasil.html',
        prediction=prediction[0], age=age, sex=sex, cp=cp, trtbps=trtbps, chol=chol, fbs=fbs,
        restecg=restecg, thalachh=thalachh, exng=exng, oldpeak=oldpeak, slp=slp, caa=caa, thall=thall
    )


if __name__ == '__main__':
    app.run(debug=True)
