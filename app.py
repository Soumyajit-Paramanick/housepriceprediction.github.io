from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('lr.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict_price():
    CRIM = float(request.form['CRIM'])
    ZN = float(request.form['ZN'])
    INDUS = float(request.form['INDUS'])
    CHAS = float(request.form['CHAS'])
    NOX = float(request.form['NOX'])
    RM = float(request.form['RM'])
    Age = float(request.form['Age'])
    DIS = float(request.form['DIS'])
    RAD =float(request.form['RAD'])
    TAX= float(request.form['TAX'])
    PTRATIO =float(request.form['PTRATIO'])
    B = float(request.form['B'])
    LSTAT= float(request.form['LSTAT'])

    result=model.predict( np.array([[CRIM,ZN,INDUS,CHAS,NOX,
               RM,Age,DIS,RAD,TAX,
               PTRATIO,B,LSTAT]]))
    print(result)

    return str(result[0])


if __name__ == '__main__':
    app.run(debug=True)