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
    return render_template('result.html')



@app.route('/predict', methods = ['POST','GET'])
def predict():
    #Rendering result on HTML GUI
    temp_array = list()
    if request.method == 'POST':
        
        team1 = request.form.get('Home-Team') #select tag name is Home-Team
        
        if team1 == '1':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
            team1= 'Mumbai Indians'
        elif team1 == '2':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
            team1= 'Sunrisers Hyderabad'
        elif team1 == '3':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
            team1= 'Kolkata Knight Riders'
        elif team1 == '4':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
            team1= 'Royal challangers banglore'
        elif team1 == '5':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
            team1= 'Delhi Capitals'
        elif team1 == '6':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
            team1= 'Chennai Super Kings'
        elif team1 == '7':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
            team1= 'Rajasthan Royals'
        elif team1 == '8':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            team1= 'Punjab Kings'
            
            
        team2 = request.form.get('Away-Team') #select tag name is Away-Team
        
        if team2 == '1':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
            team2= 'Mumbai Indians'
        elif team2 == '2':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
            team2= 'Sunrisers Hyderabad'
        elif team2 == '3':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
            team2= 'Kolkata Knight Riders'
        elif team2 == '4':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
            team2= 'Royal challangers banglore'
        elif team2 == '5':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
            team2= 'Delhi Capitals'
        elif team2 == '6':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
            team2= 'Chennai Super Kings'
        elif team2 == '7':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
            team2= 'Rajasthan Royals'
        elif team2 == '8':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            team2= 'Punjab Kings'

        input_features = [eval(x) for x in request.form.values()]
        features_values = [np.array(input_features)]

        features_name =  ['team1',  'team2', 'team1_toss_win', 'team1_bat', 'venue']

        df = pd.DataFrame(features_values, columns = features_name)

        output = model.predict(df)
        str = 'winnner'
        if output == 0.0:
            res_val = 'In the match between ' + team1 + ' and ' + team2 + ', ' + team1 +' is the ' + str
        elif output == 1.0:
            res_val = 'In the match between ' + team1 + ' and ' + team2 + ', ' + team2 +' is the ' + str

        
       
        return render_template('result.html', prediction_text = res_val)


if __name__ == '__main__':
   app.run(debug=True)

