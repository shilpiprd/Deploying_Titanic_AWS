from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pclass = request.form['pclass']
        sex = request.form['sex']
        age = request.form['age']
        fare = request.form['fare']
        embarked = request.form['embarked']
        title = request.form['title']
        famsize = request.form['famsize']
        isalone = request.form['isalone']
        
        data = [[pclass, sex, age, fare, embarked, title, famsize, isalone]]
        prediction = model.predict(data)[0]
        if prediction == 1:
            prediction = 'Survived'
        else:
            prediction = 'Not Survived'
        
        return render_template('index.html', result=prediction)

if __name__  == '__main__': 
    app.run(debug = True)