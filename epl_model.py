# Importing Libraries and Reading the Dataset

import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

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

df = pd.read_csv('fixture results.csv')

## Data Analysis and Cleaning

df['id'] = df.reset_index().index
df = df[['id', 'season', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']]
df.loc[df["result"]=='H',"winner"]=df['home_team']
df.loc[df["result"]=='A',"winner"]=df['away_team']
df.loc[df["result"]=='D',"winner"]=np.nan

#encoder
le = LabelEncoder()
df["home_team"] = le.fit_transform(df["home_team"])
df["away_team"] = le.fit_transform(df["away_team"])
df["winner"] = le.fit_transform(df["winner"].astype(str))

#outcome variable as a probability of team1 winning
df.loc[df["winner"]==df["home_team"],"home_team_win"]=1
df.loc[df["winner"]!=df["home_team"],"home_team_win"]=0

df.loc[df["result"]=='D',"draw"]=1
df.loc[df["result"]!='D',"draw"]=0

# Feature Selection

df['draw'] = df['draw'].apply(np.int64)
df['home_team'] = df['home_team'].apply(np.int64)
df['away_team'] = df['away_team'].apply(np.int64)
df['home_team_win'] = df['home_team_win'].apply(np.int64)

#dataframe of related features
prediction_df=df[['home_team', 'away_team', 'home_team_win']]

#dropping higly correlated features

crr_f = set()
crr_m = prediction_df.drop(['home_team_win'], axis=1).corr()
print('prediction_df cols are ', prediction_df.columns)
print(crr_m)

for i in range(len(crr_m.columns)): # 5
    for j in range(i): 
        print('i ', i, ' j ', j)
        if abs(crr_m.iloc[i, j]) > 0.8:
            col_name = crr_m.columns[i]
            print(col_name)
            crr_f.add(col_name)
            print('crr_f', crr_f)

#feature selection
X = prediction_df.drop(['home_team_win'] ,axis=1) # X

y = prediction_df[['home_team_win']] # y


logReg=LogisticRegression(solver='lbfgs')

print('logReg', logReg)

'''rfe = RFE(logReg, 20)

rfe = rfe.fit(X, y.values.ravel())

print('rfe', rfe)

#Checking for the features of they are important

print(rfe.support_)'''

#Splitting the data into training and testing data and scaling it

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # target is y, 
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



## Building, Training & Testing the Model

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Confusion matrix\n ", confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print('Accuracy of Logistic Regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print(confusion_matrix(y_test,y_pred),[0.0,1.0])


# SWM

svm = SVC()
svm.fit(X_train_norm,y_train)
svm.score(X_test_norm,y_test)
y1_pred = svm.predict(X_test_norm)
print("Confusion matrix\n ", confusion_matrix(y_test,y1_pred))
print(classification_report(y_test,y1_pred))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test_norm, y_test)))


# Decision Tree

dtree = DecisionTreeClassifier()
dtree.fit(X_train_norm,y_train)
dtree.score(X_test_norm,y_test)
y2_pred = dtree.predict(X_test_norm)
print("Confusion matrix\n ", confusion_matrix(y_test,y2_pred))
print(classification_report(y_test,y2_pred))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(dtree.score(X_test_norm, y_test)))


# Random Forest 

randomForest= RandomForestClassifier(n_estimators=100)
randomForest.fit(X_train_norm,y_train)
randomForest.score(X_test_norm,y_test)
y3_pred = randomForest.predict(X_test_norm)
print("Confusion matrix\n ", confusion_matrix(y_test,y3_pred))
print(classification_report(y_test,y3_pred))
print('Accuracy of Random Forest classifier on test set: {:.4f}'.format(randomForest.score(X_test_norm, y_test)))



# Creating a pickle file for the classifier
import pickle

'''filename = 'EPL-match-prediction-lr-model.pkl'
pickle.dump(logreg, open(filename, 'wb'))'''

# Naive Bayes

naivebayes_classifier = GaussianNB()
naivebayes_classifier.fit(X_train,y_train)
naivebayes_classifier.score(X_test,y_test)
y4_pred  =  naivebayes_classifier.predict(X_test)
print("Confusion matrix\n ", confusion_matrix(y_test,y4_pred))
print(classification_report(y_test,y4_pred))
print('Accuracy of Naive Bayes classifier on test set: {:.4f}'.format(naivebayes_classifier.score(X_test, y_test)))