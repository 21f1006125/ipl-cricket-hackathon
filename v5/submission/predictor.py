#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def predictRuns(testInput):
    prediction = 0
    ### Your Code Here ###
    with open('.//ipl_csv2//all_matches.csv') as f:
        #ipl_data = pd.read_csv(f, low_memory=False)
        ipl_data = pd.read_csv(f, dtype={"match_id":int,"season":"string","start_date":"string","venue":"string",
                                         "innings":int,"ball":float,"batting_team":"string","bowling_team":"string",
                                         "striker":"string","non_striker":"string","bowler":"string","runs_off_bat":int,
                                         "extras":int,"wides":pd.Int64Dtype(),"noballs":pd.Int64Dtype(),
                                         "byes":pd.Int64Dtype(),"legbyes":pd.Int64Dtype(),"penalty":pd.Int64Dtype(),
                                         "wicket_type":"string","player_dismissed":"string",
                                         "other_wicket_type":"string","other_player_dismissed":"string"})
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
    ipl_data['venue'] = ipl_data['venue'].replace(['Sardar Patel Stadium, Motera'],'Narendra Modi Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['Punjab Cricket Association IS Bindra Stadium, Mohali'],'Punjab Cricket Association IS Bindra Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['Punjab Cricket Association Stadium, Mohali'],'Punjab Cricket Association IS Bindra Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['Wankhede Stadium, Mumbai'],'Wankhede Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['Feroz Shah Kotla'],'Arun Jaitley Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['MA Chidambaram Stadium, Chepauk, Chennai'],'MA Chidambaram Stadium')
    ipl_data['venue'] = ipl_data['venue'].replace(['MA Chidambaram Stadium, Chepauk'],'MA Chidambaram Stadium')
    venue_encoder = LabelEncoder()
    team_encoder = LabelEncoder()
    player_encoder = LabelEncoder()
    ipl_data['venue'] = venue_encoder.fit_transform(ipl_data['venue'])
    ipl_data['batting_team'] = team_encoder.fit_transform(ipl_data['batting_team'])
    ipl_data['bowling_team'] = team_encoder.fit_transform(ipl_data['bowling_team'])
    anArray = ipl_data.to_numpy()
    anArray = anArray.astype(int)
    X,y = anArray[:,:3], anArray[:,4]
    X = np.concatenate((np.eye(42)[anArray[:,0]], np.eye(2)[anArray[:,1]-1], np.eye(15)[anArray[:,2]], np.eye(15)[anArray[:,3]],), axis = 1)
    X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)
    clf = RandomForestClassifier(n_estimators=100, random_state=None)
    #clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    #linearRegressor = LinearRegression()
    #linearRegressor.fit(X_train, y_train)
    test_case = pd.read_csv(testInput)
    batsmen = test_case['batsmen'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('batsmen')
    bowlers = test_case['bowlers'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('bowlers')
    test_case = test_case.drop('batsmen', axis=1).join(batsmen)
    test_case = test_case.drop('bowlers', axis=1).join(bowlers)
    test_case['venue'] = venue_encoder.transform(test_case['venue'])
    test_case['batting_team'] = team_encoder.transform(test_case['batting_team'])
    test_case['bowling_team'] = team_encoder.transform(test_case['bowling_team'])
    test_case['batsmen'] = player_encoder.transform(test_case['batsmen'])
    test_case['bowlers'] = player_encoder.transform(test_case['bowlers'])
    #test_case = test_case[['venue', 'innings', 'batting_team', 'bowling_team']]
    testArray = test_case.to_numpy()
    test_case = np.concatenate((np.eye(42)[testArray[:,0]], np.eye(2)[testArray[:,1]-1], np.eye(15)[testArray[:,2]], np.eye(15)[testArray[:,3]],), axis = 1)
    #prediction = linearRegressor.predict(test_case)
    prediction = clf.predict(test_case)
    return round(prediction[0])


# In[ ]:




