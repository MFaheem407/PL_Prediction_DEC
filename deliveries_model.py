# Importing libraries and csv

import pandas as pd
import numpy as np
import math
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

df_raw = pd.read_csv('deliveries.csv')

# ### Data Analysis and Cleaning
df_raw["season"] = np.nan

## Feature Generation

# ### Feature Engineering

cols = list(df_raw.columns)
cols = [cols[-1]] + cols[:-1]
df_raw = df_raw[cols]

df_raw.loc[(df_raw['match_id']<=59)  & (df_raw['match_id']>=1), 'season']     = 2017
df_raw.loc[(df_raw['match_id']<=117) & (df_raw['match_id']>=60), 'season']    = 2008
df_raw.loc[(df_raw['match_id']<=174) & (df_raw['match_id']>=118), 'season']   = 2009
df_raw.loc[(df_raw['match_id']<=234) & (df_raw['match_id']>=175), 'season']   = 2010
df_raw.loc[(df_raw['match_id']<=307) & (df_raw['match_id']>=235), 'season']   = 2011
df_raw.loc[(df_raw['match_id']<=381) & (df_raw['match_id']>=308), 'season']   = 2012
df_raw.loc[(df_raw['match_id']<=457) & (df_raw['match_id']>=382), 'season']   = 2013
df_raw.loc[(df_raw['match_id']<=517) & (df_raw['match_id']>=458), 'season']   = 2014
df_raw.loc[(df_raw['match_id']<=576) & (df_raw['match_id']>=518), 'season']   = 2015
df_raw.loc[(df_raw['match_id']<=636) & (df_raw['match_id']>=577), 'season']   = 2016
df_raw.loc[(df_raw['match_id']<=7953) & (df_raw['match_id']>=7894), 'season'] = 2018
df_raw.loc[(df_raw['match_id']<=11415) & (df_raw['match_id']>=11137), 'season'] = 2019

df_raw.loc[(df_raw['dismissal_kind']=='caught') | 
           (df_raw['dismissal_kind']=='bowled') |
           (df_raw['dismissal_kind']=='run out') |
           (df_raw['dismissal_kind']=='lbw') |
           (df_raw['dismissal_kind']=='stumped') |
           (df_raw['dismissal_kind']=='caught and bowled') |
           (df_raw['dismissal_kind']=='hit wicket') |
           (df_raw['dismissal_kind']=='retired hurt') |
           (df_raw['dismissal_kind']=='obstructing the field'), 'wicket'] = 1

df_raw['wicket'] = df_raw['wicket'].fillna(0)
df_raw['wicket'] = df_raw['wicket'].apply(np.int64)

L =[2008,2009, 2010, 2011, 2012, 2013, 2014, 2015,2016,2017, 2018, 2019]
df_raw['season'] = pd.Categorical(df_raw['season'], ordered=True, categories=L)

df = df_raw.sort_values(['season', 'match_id'], ignore_index=True)

# shift column 'match_id' to first position
first_column = df.pop('match_id')
  
# insert column using insert(position,column_name,
# first_column) function
df.insert(0, 'match_id', first_column)

df['season'] = df['season'].astype(str).astype(np.int64)


# these teams names were changed over the years.
df = df.replace(['Deccan Chargers', 'Delhi Daredevils', 'Rising Pune Supergiants', 'Kings XI Punjab'], 
                ['Sunrisers Hyderabad', 'Delhi Capitals', 'Rising Pune Supergiant', 'Punjab Kings'])

#df['id'] = df['match_id']

#df.set_index('id', inplace = True)


df['runs_last_5'] = 0
df['wickets_last_5'] = 0
df['innings_score'] = 0

df['over']=df['over']-1


df['over']=df['over'].astype(str)
df['ball']=df['ball'].astype(str)



df['over/ball']=df['over']+'.'+df['ball']

columns_to_remove = ['over', 'ball']
df.drop(labels=columns_to_remove, axis=1, inplace=True)

df['over/ball'] = df['over/ball'].astype(float)

df['sum_total_runs'] = df.groupby(['season', 'match_id', 'inning'])['total_runs'].cumsum()
df['sum_total_wickets'] = df.groupby(['season', 'match_id', 'inning'])['wicket'].cumsum()

df = df[['match_id', 'season', 'inning', 'over/ball', 'batting_team', 'bowling_team',
       'batsman', 'non_striker', 'bowler', 'is_super_over', 'wide_runs',
       'bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs',
       'batsman_runs', 'extra_runs', 'total_runs', 'sum_total_runs', 'player_dismissed',
       'dismissal_kind', 'wicket', 'sum_total_wickets', 'fielder', 'runs_last_5', 'wickets_last_5', 'innings_score']]

df = df.rename({'penalty_runs': 'freehit_runs'}, axis=1)

grouping_variables = ['season', 'match_id', 'inning']
df = df.set_index(grouping_variables)
df = df.assign(innings_score=df.groupby(grouping_variables)["sum_total_runs"].max())
df = df.reset_index()
df = df.rename({'total_runs': 'total_runs_per_ball'}, axis=1)
df = df[['season', 'match_id', 'inning', 'batting_team', 'bowling_team',
       'over/ball', 'total_runs_per_ball', 'sum_total_runs', 'sum_total_wickets', 'runs_last_5', 'wickets_last_5', 'innings_score']]

unq_seasons = list(df['season'].unique())
unq_innings = [1,2]

df['over/ball'] = df['over/ball'].astype(str)

for season in unq_seasons:
    unq_match_ids = list(df[df['season'] == season]['match_id'].unique())
    for match_id in unq_match_ids:
        for inning in unq_innings:
            mini_df = df.loc[(df['season'] == season) & (df['match_id'] == match_id) & (df['inning'] == inning),:]
            total_overs_ball = mini_df['over/ball'].tolist()
            total_wicket_ball = mini_df['sum_total_wickets'].tolist()
            total_runs = mini_df['sum_total_runs'].tolist()
            total_ind = list(mini_df.index)
            mini_df = mini_df.set_index('over/ball')
            wic_last_5 = []
            run_last_5 = []
            for i,(tob, ti) in enumerate(zip(total_overs_ball, total_ind)):
                curr_over = int(tob.split('.')[0])
                curr_ball = int(tob.split('.')[1])
                
                prev_over = int(curr_over - 5)
                prev_ball = curr_ball
                if curr_ball not in range(1,7):
                    prev_over = int(curr_over - 4) 
                    prev_ball = int(curr_ball %6)
                try:
                    mini_df.loc[f"{prev_over}.{prev_ball}"]
                    t_w = mini_df.loc[f"{prev_over}.{prev_ball}": f"{curr_over}.{curr_ball}"]['sum_total_wickets'].tolist()
                    t_r = mini_df.loc[f"{prev_over}.{prev_ball}": f"{curr_over}.{curr_ball}"]['sum_total_runs'].tolist()
                    tw_5 = int(t_w[-1]) - int(t_w[0])
                    tr_5 = int(t_r[-1]) - int(t_r[0])
                    wic_last_5.append(tw_5)
                    run_last_5.append(tr_5)
                except Exception as e:
                    wic_last_5.append(int(total_wicket_ball[i]))
                    run_last_5.append(int(total_runs[i]))
            if len(total_ind) != len(wic_last_5):
                print("Invalid Data")
            df.loc[total_ind,'wickets_last_5'] = wic_last_5
            df.loc[total_ind,'runs_last_5'] = run_last_5



# #### Feature Selection

df['over/ball'] = df['over/ball'].astype(float)
df['wickets_last_5'] = df['wickets_last_5'].astype(int)
df['runs_last_5'] = df['runs_last_5'].astype(int)
# Removing the first 5 overs data in every match
df = df[df['over/ball']>=5.0]
df = df[df['inning'] == 1]

# --- Data Preprocessing ---
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['batting_team', 'bowling_team'])

# Rearranging the columns
encoded_df = encoded_df[['season', 'batting_team_Chennai Super Kings', 'batting_team_Delhi Capitals', 'batting_team_Punjab Kings',
              'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
              'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad',
              'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Capitals', 'bowling_team_Punjab Kings',
              'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals',
              'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad',
              'over/ball', 'sum_total_runs', 'sum_total_wickets', 'runs_last_5', 'wickets_last_5', 'innings_score']]


# Splitting the data into train and test set
X_train = encoded_df.drop(labels='innings_score', axis=1)[encoded_df['season'] <= 2018]
X_test = encoded_df.drop(labels='innings_score', axis=1)[encoded_df['season'] >= 2019]

y_train = encoded_df[encoded_df['season'] <= 2018]['innings_score'].values
y_test = encoded_df[encoded_df['season'] >= 2019]['innings_score'].values

# Removing the 'date' column
X_train.drop(labels='season', axis=True, inplace=True)
X_test.drop(labels='season', axis=True, inplace=True)


## Building, Training & Testing the Model

# #### Linear Regression Model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set result  
y_pred= regressor.predict(X_test) 

print("lr.coef_: {}".format(regressor.coef_))
print("lr.intercept_: {}".format(regressor.intercept_))
print("Training set score: {:.2f}".format(regressor.score(X_train, y_train)))
print("Test set score: {:.7f}".format(regressor.score(X_test, y_test)))

print('Accuracy of Linear Regression classifier on test set: {:.2f}'.format(regressor.score(X_test, y_test)))


# Creating a pickle file for the classifier

'''filename = 'first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))'''



## Ridge Regression

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)


#Predicting the test set result  
y1_pred= ridge_regressor.predict(X_test) 

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

prediction=ridge_regressor.predict(X_test)
print(prediction)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

print('Accuracy of Ridge Regression classifier on test set: {:.2f}'.format(ridge_regressor.score(X_test, y_test)))

## Lasso Regression

from sklearn.linear_model import Lasso

lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

#Predicting the test set result  
y2_pred= lasso_regressor.predict(X_test) 