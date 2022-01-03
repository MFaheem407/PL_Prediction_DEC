from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the Logistic Regression model

filename = 'IPL-match-prediction-lr-model.pkl'

model = pickle.load(open(filename, 'rb'))

@app.route('/LandingPage.html')
def home_page():
    return render_template('LandingPage.html')

@app.route('/stats.html')
def stats_page():
    return render_template('stats.html')

@app.route('/result.html')
def result_page():
    print('team1')
    return render_template('result.html')



@app.route('/predict', methods = ['POST','GET'])
def predict():
    print('team222222')
    #Rendering result on HTML GUI
    temp_array = list()
    if request.method == 'POST':
        
        team1 = request.form.get('batting-team')
        print("Team1 is",team1)
        if team1 == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif team1 == 'Delhi Daredevils':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif team1 == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif team1 == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif team1 == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif team1 == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif team1 == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif team1 == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
            
        team2 = request.form.get('bowling-team')
        print("Team2 is", team2)
        if team2 == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif team2 == 'Delhi Daredevils':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif team2 == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif team2 == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif team2 == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif team2 == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif team2 == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif team2 == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]

        input_features = [int(x) for x in request.form.values()]
        features_values = [np.array(input_features)]

        features_name =  ['team1',  'team2', 'team1_wins']

        df = pd.DataFrame(features_values, columns = features_name)

        output = model.predict(df)
        str1 = 'wins'
        if output == 0.0:
            res_val = team1+str1
        elif output == 1.0:
            res_val = team2+str1

        return render_template('result.html', prediction_text = '{}'.format(output))


if __name__ == '__main__':
   app.run(debug=True)

