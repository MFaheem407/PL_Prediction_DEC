# Importing libraries and csv

import pandas as pd
import numpy as np
import math
import pickle
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
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler




raw_df = pd.read_csv("matches.csv")


df = raw_df.replace(['Deccan Chargers', 'Delhi Daredevils', 'Rising Pune Supergiants'],                                      
                    ['Sunrisers Hyderabad', 'Delhi Capitals', 'Rising Pune Supergiant'])



# Filling the values of city based on venue

conditions = [df["venue"] == "Rajiv Gandhi International Stadium, Uppal",
              df["venue"] == "Maharashtra Cricket Association Stadium",
              df["venue"] == "Saurashtra Cricket Association Stadium", 
              df["venue"] == "Holkar Cricket Stadium",
              df["venue"] == "M Chinnaswamy Stadium",
              df["venue"] == "Wankhede Stadium",
              df["venue"] == "Eden Gardens",
              df["venue"] == "Feroz Shah Kotla",
              df["venue"] == "Punjab Cricket Association IS Bindra Stadium, Mohali",
              df["venue"] == "Green Park",
              df["venue"] == "Punjab Cricket Association Stadium, Mohali",
              df["venue"] == "Dr DY Patil Sports Academy",
              df["venue"] == "Sawai Mansingh Stadium", 
              df["venue"] == "MA Chidambaram Stadium, Chepauk", 
              df["venue"] == "Newlands", 
              df["venue"] == "St George's Park" , 
              df["venue"] == "Kingsmead", 
              df["venue"] == "SuperSport Park",
              df["venue"] == "Buffalo Park", 
              df["venue"] == "New Wanderers Stadium",
              df["venue"] == "De Beers Diamond Oval", 
              df["venue"] == "OUTsurance Oval", 
              df["venue"] == "Brabourne Stadium",
              df["venue"] == "Sardar Patel Stadium", 
              df["venue"] == "Barabati Stadium", 
              df["venue"] == "Vidarbha Cricket Association Stadium, Jamtha",
              df["venue"] == "Himachal Pradesh Cricket Association Stadium",
              df["venue"] == "Nehru Stadium",
              df["venue"] == "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
              df["venue"] == "Subrata Roy Sahara Stadium",
              df["venue"] == "Shaheed Veer Narayan Singh International Stadium",
              df["venue"] == "JSCA International Stadium Complex",
              df["venue"] == "Sheikh Zayed Stadium",
              df["venue"] == "Sharjah Cricket Stadium",
              df["venue"] == "Dubai International Cricket Stadium",
              df["venue"] == "M. A. Chidambaram Stadium",
              df["venue"] == "Feroz Shah Kotla Ground",
              df["venue"] == "M. Chinnaswamy Stadium",
              df["venue"] == "Rajiv Gandhi Intl. Cricket Stadium" ,
              df["venue"] == "IS Bindra Stadium",
              df["venue"] == "ACA-VDCA Stadium"]

values = ['Hyderabad', 'Mumbai', 'Rajkot',"Indore","Bengaluru","Mumbai","Kolkata","Delhi","Mohali","Kanpur","Mohali","Pune","Jaipur","Chennai","Cape Town","Port Elizabeth","Durban",
          "Centurion",'Eastern Cape','Johannesburg','Northern Cape','Bloemfontein','Mumbai','Ahmedabad','Cuttack','Jamtha','Dharamshala','Chennai','Visakhapatnam','Pune','Raipur','Ranchi',
          'Abu Dhabi','Sharjah','Dubai','Chennai','Delhi','Bengaluru','Hyderabad','Mohali','Visakhapatnam']

df['city'] = np.where(df['city'].isnull(),
                              np.select(conditions, values),
                              df['city'])




# Removing records having null values in "winner" column

df = df[df["winner"].notna()]


# Feature Generation

###Feature Engineering

# encoder

le = LabelEncoder()
df["team1"] = le.fit_transform(df["team1"])
df["team2"] = le.fit_transform(df["team2"])
df["winner"] = le.fit_transform(df["winner"].astype(str))
df["toss_winner"] = le.fit_transform(df["toss_winner"])
df["venue"] = le.fit_transform(df["venue"])


# outcome variable as a probability of team1 winning

df.loc[df["winner"]==df["team1"],"team1_win"]=1
df.loc[df["winner"]!=df["team1"],"team1_win"]=0

df.loc[df["toss_winner"]==df["team1"],"team1_toss_win"]=1
df.loc[df["toss_winner"]!=df["team1"],"team1_toss_win"]=0

df["team1_bat"]=0
df.loc[(df["team1_toss_win"]==1) & (df["toss_decision"]=="bat"),"team1_bat"]=1


## Feature Selection



# dataframe of related features

df['team1'] = df['team1'].astype(str).astype(int)
df['team2'] = df['team2'].astype(str).astype(int)
df['venue'] = df['venue'].astype(str).astype(int)

# dataframe of related features

prediction_df=df[["team1","team2","team1_toss_win","team1_bat","team1_win","venue"]]


# dropping higly correlated features

crr_f = set()
crr_m = prediction_df.drop(['team1_win'], axis=1).corr()


for i in range(len(crr_m.columns)): # 5
    for j in range(i): 
        #print('i ', i, ' j ', j)
        if abs(crr_m.iloc[i, j]) > 0.8:
            col_name = crr_m.columns[i]
            #print(col_name)
            crr_f.add(col_name)
            #print('crr_f', crr_f)


# feature selection
X = prediction_df.drop('team1_win', axis=1) # X
target = prediction_df['team1_win'] # target is y


logReg=LogisticRegression(solver='lbfgs')

'''rfe = RFE(logReg, 20) # 20 is test percentage

rfe = rfe.fit(X, target.values.ravel())'''

# Splitting the data into training and testing data and scaling it

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=0) # target is y, 
# test size is telling that 20% is test data

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# transform training data
X_train_norm = norm.transform(X_train)

# transform testing dataabs
X_test_norm = norm.transform(X_test)

print('X.shape', X.shape)
print('X_train.shape', X_train.shape)
print('X_test.shape', X_test.shape)


# ## Building, Training & Testing the Model

# ### Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Confusion matrix\n ", confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print('Accuracy of Logistic Regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print(confusion_matrix(y_test,y_pred),[0.0,1.0])

# Creating Pickle file
'''filename = 'IPL-match-prediction-lr-model.pkl'
pickle.dump(logreg, open(filename, 'wb'))'''

# ### SWM

svm = SVC()
svm.fit(X_train_norm,y_train)
svm.score(X_test_norm,y_test)
y1_pred = svm.predict(X_test_norm)
print("Confusion matrix\n ", confusion_matrix(y_test,y1_pred))
print(classification_report(y_test,y1_pred))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test_norm, y_test)))

# ### Decision Tree

dtree = DecisionTreeClassifier()
dtree.fit(X_train_norm,y_train)
dtree.score(X_test_norm,y_test)
y2_pred = dtree.predict(X_test_norm)
print("Confusion matrix\n ", confusion_matrix(y_test,y2_pred))
print(classification_report(y_test,y2_pred))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(dtree.score(X_test_norm, y_test)))

# ### Random Forest

randomForest= RandomForestClassifier(n_estimators=100)
randomForest.fit(X_train_norm,y_train)
randomForest.score(X_test_norm,y_test)
y3_pred = randomForest.predict(X_test_norm)
print("Confusion matrix\n ", confusion_matrix(y_test,y3_pred))
print(classification_report(y_test,y3_pred))
print('Accuracy of Random Forest classifier on test set: {:.4f}'.format(randomForest.score(X_test_norm, y_test)))

# ### Naive Bayes

naivebayes_classifier = GaussianNB()
naivebayes_classifier.fit(X_train,y_train)
naivebayes_classifier.score(X_test,y_test)
y4_pred  =  naivebayes_classifier.predict(X_test)
print("Confusion matrix\n ", confusion_matrix(y_test,y4_pred))
print(classification_report(y_test,y4_pred))
print('Accuracy of Naive Bayes classifier on test set: {:.4f}'.format(naivebayes_classifier.score(X_test, y_test)))

