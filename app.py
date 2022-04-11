from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', BMI=0, Status_Gender=2)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
     if request.method == "POST":
        #get form data
        Status_Gender = request.form.get('Status_Gender')
        Height = request.form.get('Height')  
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(Status_Gender, Height)
            #pass prediction to template
            return render_template('index.html', Weight=prediction, Status_Gender=Status_Gender, Height=Height)
        except ValueError:
            return "Please Enter valid values"
pass
pass

def preprocessDataAndPredict(Status_Gender, Height):
    #keep all inputs in array
    test_data = [Status_Gender, Height]
    print(test_data)
    #convert value data into numpy array
    test_data = np.array(test_data)
    #reshape array
    test_data = test_data.reshape(1,-1)
    print(test_data)
    #open file
    file = open("modelbmi.sav","rb")
    #load trained model
    trained_model = joblib.load(file)
    #predict
    prediction = trained_model.predict(test_data)
    return prediction
pass

if __name__ == '__name__':
    app.run(debug=True)