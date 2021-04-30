#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

def predictRuns(testInput):
    prediction = 0
    ### Your Code Here ###
    with open('.//all_matches.csv') as f:
        ipl_data = pd.read_csv(f)
    relevantColumns = ['match_id', 'venue', 'innings', 'ball', 'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler','runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes','penalty']
    ipl_data = ipl_data[relevantColumns]
    ipl_data['total_runs'] = ipl_data['runs_off_bat'] + ipl_data['extras']
    ipl_data = ipl_data.drop(columns=['runs_off_bat','extras'])
    ipl_data = ipl_data[ipl_data['ball'] <= 5.6]
    ipl_data = ipl_data[ipl_data['innings'] <= 2]
    ipl_data = ipl_data.groupby(['match_id', 'venue', 'innings', 'batting_team', 'bowling_team' ]).total_runs.sum()
    ipl_data = ipl_data.reset_index()
    ipl_data = ipl_data.drop(columns=['match_id'])
    ipl_data['batting_team'] = ipl_data['batting_team'].replace(['Kings XI Punjab'],'Punjab Kings')
    ipl_data['bowling_team'] = ipl_data['bowling_team'].replace(['Kings XI Punjab'],'Punjab Kings')
    ipl_data['bowling_team'] = ipl_data['bowling_team'].replace(['Delhi Daredevils'],'Delhi Capitals')
    ipl_data['batting_team'] = ipl_data['batting_team'].replace(['Delhi Daredevils'],'Delhi Capitals')
    ipl_data['venue'] = ipl_data['venue'].replace(['M Chinnaswamy Stadium'],'M.Chinnaswamy Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['Rajiv Gandhi International Stadium, Uppal'],'Rajiv Gandhi International Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['Sardar Patel Stadium, Moter'],'Narendra Modi Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['Punjab Cricket Association IS Bindra Stadium, Mohali'],'Punjab Cricket Association IS Bindra Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['Punjab Cricket Association Stadium, Mohali'],'Punjab Cricket Association IS Bindra Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['Wankhede Stadium, Mumbai'],'Wankhede Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['Feroz Shah Kotla'],'Arun Jaitley Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['MA Chidambaram Stadium, Chepauk, Chennai'],'MA Chidambaram Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['MA Chidambaram Stadium, Chepauk'],'MA Chidambaram Stadium')
    venue_encoder = LabelEncoder()
    team_encoder = LabelEncoder()
    ipl_data['venue'] = venue_encoder.fit_transform(ipl_data['venue'])
    ipl_data['batting_team'] = team_encoder.fit_transform(ipl_data['batting_team'])
    ipl_data['bowling_team'] = team_encoder.fit_transform(ipl_data['bowling_team'])
    anArray = ipl_data.to_numpy()
    X,y = anArray[:,:3], anArray[:,4]
    X = np.concatenate((np.eye(42)[anArray[:,0]], np.eye(2)[anArray[:,1]-1], np.eye(15)[anArray[:,2]], np.eye(15)[anArray[:,3]],), axis = 1)
    X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)
    linearRegressor = LinearRegression()
    linearRegressor.fit(X_train, y_train)
    test_case = pd.read_csv(testInput)
    test_case['venue'] = venue_encoder.transform(test_case['venue'])
    test_case['batting_team'] = team_encoder.transform(test_case['batting_team'])
    test_case['bowling_team'] = team_encoder.transform(test_case['bowling_team'])
    test_case = test_case[['venue', 'innings', 'batting_team', 'bowling_team']]
    testArray = test_case.to_numpy()
    test_case = np.concatenate((np.eye(42)[testArray[:,0]], np.eye(2)[testArray[:,1]-1], np.eye(15)[testArray[:,2]], np.eye(15)[testArray[:,3]],), axis = 1)
    prediction = linearRegressor.predict(test_case)
    return round(prediction[0])


# In[ ]:




