import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
model= load('Multyy.save')
trans=load('Transformss')

@app.route('/')
def home():
    return render_template('home.html')
    

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    print(x_test)
    test=trans.transform(x_test)
    test=test[:,0:]
    print(test)
    prediction = model.predict(test) 
    print(prediction)
    output=prediction[0] 
    
    return render_template('index.html', prediction_text='Concrete compressive strength(MPa, megapascals)  {}'.format(output))

'''@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    #For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True)
