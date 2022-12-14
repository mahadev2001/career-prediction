from flask import Flask,request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html',pred='Sutaible career option for this student is {}'.format(predclassprob))
    if request.method == 'POST':
        for x in request.form.values():
            print(x)
            print("value")
        userdata=[int(x) for x in request.form.values()]
        final=[np.array(userdata)]
        print(model.predict(userdata)) 
        classprobs = model.predict_proba(userdata)
        predclassprob = np.max(classprobs)

        return render_template('result.html',result=predclassprob)
  
if __name__ == '__main__':
    app.run(debug=True)
