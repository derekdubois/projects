import os
os.environ['SQLALCHEMY_SILENCE_UBER_WARNING'] = '1'
import requests
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from imblearn.over_sampling import SMOTE
from joblib import dump



# initial fetch code
# def get_data(url):
#     response = requests.get(url)
#     #json = response.json()
#     #df = pd.DataFrame(json)
#     return response.json()


# def fetch_player_history(player_id):
#     # API URL
#     player_url = f'https://fantasy.premierleague.com/api/element-summary/{player_id}/'

#     # get player data
#     player_data = get_data(player_url)

#     #extract historical data
#     history_df = pd.DataFrame(player_data['history'])

#     # filter out unnecessary columns
#     history_df = history_df[['round', 'expected_goal_involvements', 'ict_index', 'goals_scored', 'assists']]

#     return history_df

## initial fetch from API
# # get elements df
# url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
# data = get_data(url)
# elements_df = pd.DataFrame(data['elements'])

# #get list of player IDs
# player_ids = elements_df['id'].values

# #initialize dictionary to hold player data
# player_data = {}

# #fetch data for players
# #for player_id in player_ids:
#  #   print(f"Fetching data for player {player_id}...")
#   #  player_data[player_id] = fetch_player_history(player_id)

# #with open('player_data.pickle_2', 'wb') as handle:
#  #   pickle.dump(player_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# with open('player_data_2.pickle', 'rb') as handle:
#     player_data = pickle.load(handle)


#CFI calculate function
#def calculate_cfi(player_data):
  #  for player_id, history_data in player_data.items():
   #     df = pd.DataFrame(history_data)
    #    df = df[['round', 'expected_goal_involvements', 'ict_index', 'goals_scored', 'assists']]

        #order the dataframe by gameweek, most recent game last
     #   df.sort_values('round', inplace=True)

        #check if dataframe is empty or if the player has scored or assisted in the last four gameweeks
      #  if df.empty or df.iloc[-4:][['goals_scored', 'assists']].sum().sum() > 0 or len(df) < 4:
       #     player_data[player_id]['CFi'] = 0
        #else:
         #   gW1_xG = df.iloc[-1]['expected_goal_involvements']
          #  gW2_xG = df.iloc[-2]['expected_goal_involvements']
           # gW3_xG = df.iloc[-3]['expected_goal_involvements']
            #gW4_xG = df.iloc[-4]['expected_goal_involvements']
            #iCT = df.iloc[-1]['ict_index']

            #CFi = (gW4_xG * 0.7) + (gW3_xG * 0.8) + (gW2_xG * 0.9) + gW1_xG + (iCT * 0.001)
            #player_data[player_id]['CFi'] = CFi

    #return player_data

# def calculate_cfi(player_df):
#     # if player_df is a dictionary, convert to dataframe
#     if isinstance(player_df, dict):
#         df = pd.DataFrame(player_df).T
#     else:
#         df = player_df.copy()

#     #print columns for debugging
#     #print(df.columns)

#     df = df[['round', 'expected_goal_involvements', 'ict_index', 'goals_scored', 'assists']]

#     #skip if not enough data
#     if df.empty or df['goals_scored'].sum() > 0 or df['assists'].sum() > 0 or len(df) < 4:
#         return None

#     #calculate CFi
#     df['xG'] = df['expected_goal_involvements'].astype(float)
#     df['iCT'] = df['ict_index'].astype(float)
#     gW4_xG = df.iloc[-4]['xG'] if len(df) >= 4 else 0
#     gW3_xG = df.iloc[-3]['xG'] if len(df) >= 3 else 0
#     gW2_xG = df.iloc[-2]['xG'] if len(df) >= 2 else 0
#     gW1_xG = df.iloc[-1]['xG']
#     gW1_iCT = df.iloc[-1]['iCT']
#     CFi = ((gW4_xG * 0.7) + (gW3_xG * 0.8) + (gW2_xG * 0.9) + gW1_xG) + (gW1_iCT * .001)

#     return CFi


# clean the data code
# # Fetch all player data
# url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
# response = requests.get(url)
# data = response.json()

# # extract the player information from the response
# players = data['elements']

# # prepare a dictionary to hold our data
# player_data = {}

# # loop over the players
# for player in players:
#     #skip goalkeepers
#     if player['element_type'] == 1:
#         continue
#     # for the remaining players, fetch their detailed info
#     response = requests.get(f'https://fantasy.premierleague.com/api/element-summary/{player["id"]}/')
#     player_info = response.json()

#     # extract the info for each round and add it to the dictionary
#     for round_info in player_info['history']:
#         round_number = round_info['round']
#         player_data.setdefault(player['id'], {}).setdefault(round_number, {
#             'round': round_number,
#             'expected_goal_involvements': round_info['expected_goal_involvements'],
#             'ict_index': round_info['ict_index'],
#             'goals_scored': round_info['goals_scored'],
#             'assists': round_info['assists'],
#             'player_name': player['web_name'],
#             'team': player['team'],
#             'position': player['element_type']
#         })
#     #at this point, 'player_data' is a dictionary where the key is the player id and the value is a DataFrame containing player's info and historical data

# # save the data
# with open('player_data.pkl', 'wb') as f:
#     pickle.dump(player_data, f)

#load the data
# with open('player_data.pkl', 'rb') as f:
#     player_data = pickle.load(f)

# for player_id, player_dict in player_data.items():
#   player_df = pd.DataFrame(player_dict) # convert dictionary to DataFrame

#   print(player_df.columns)
#   print(player_df.head())

#   # Transpose the dataframe
#   player_df = player_df.transpose()

#   # convert round from index to column
#   player_df.reset_index(level=0, inplace=True)
#   player_df.rename(columns={'index':'round'}, inplace=True)

#   # convert round to numeric
#   player_df['round'] = pd.to_numeric(player_df['round'], errors='coerce')

#   # get expected goal involvements (xG) for the previous weeks
#   player_df["gW4_xG"] = player_df["expected_goal_involvements"].shift(4).fillna(0)
#   player_df["gW3_xG"] = player_df["expected_goal_involvements"].shift(3).fillna(0)
#   player_df["gW2_xG"] = player_df["expected_goal_involvements"].shift(2).fillna(0)
#   player_df["gW1_xG"] = player_df["expected_goal_involvements"].shift(1).fillna(0)

#   # compute CFi
#   player_df["CFi"] = (player_df["gW4_xG"] * 0.7) + (player_df["gW3_xG"] * 0.8) + \
#                    (player_df["gW2_xG"] * 0.9) + player_df["gW1_xG"] + (player_df["ict_index"] * 0.001)

#   # remove temporary xG columns

#   player_df.drop(columns=["gW4_xG", "gW3_xG", "gW2_xG", "gW1_xG"], inplace=True)

#   # add in next game goals and assist columns
#   player_df["next_game_goals"] = player_df["goals_scored"].shift(-1)
#   player_df["next_game_assists"] = player_df["assists"].shift(-1)

#   # drop the last row
#   player_df = player_df.iloc[:-1]

#   # update player data in dictionary
#   player_data[player_id] = player_df

# print(player_df.head())


# # print the data for the first 3 players
# for i, (player_id, player_dict) in enumerate(player_data.items()):
#     if i > 2: # only print for first 3 players
#         break

#     print(f"Player ID: {player_id}")
#     player_df = pd.DataFrame(player_dict).transpose() #convert dict to DataFrame
#     print(player_df.head())

# #compute CFi for each player for each round
# for player_id, player_dict in player_data.items():
#     player_df = pd.DataFrame(player_dict) #convert dictionary to DataFrame



#     #convert data types
#     numeric_columns = ["expected_goal_involvements", "ict_index", "goals_scored", "assists"]
#     for column in numeric_columns:
#         if column in player_df.columns:
#             player_df[column] = player_df[column].astype(float)
#             player_df[column] = pd.to_numeric(player_df[column], errors='coerce')


# SQL code
# # connection to SQLite
# conn = sqlite3.connect('fantasy_football.db')

# # loop over data dict
# for player_id, df in data.items():
#     # add player_id column to the DataFrame
#     df['player_id'] = player_id
#     # write the DataFrame to SQL database
#     df.to_sql('gameweeks', conn, if_exists='append', index=False)

# conn.close()




# # check data type
# print(type(data))

# # if its a list or dict, check length or number of keys
# if isinstance(data, list):
#     print(len(data))
# elif isinstance(data, dict):
#     print(len(data.keys()))







# SQL DATABASE
# # establish connection to SQLite database
# conn = sqlite3.connect('fantasy_football.db')


# # create a cursor object to execute SQL commands
# cur = conn.cursor()

# # delete all rows from the gameweeks table
# cur.execute("DELETE FROM gameweeks")

# # set API endpoint
# url = "https://fantasy.premierleague.com/api/bootstrap-static/"

# # send GET response to API
# response = requests.get(url)

# # parse JSON response
# data = json.loads(response.text)

# # extract players data
# players_data = data['elements']

# # loop over players data
# for player in players_data:

#   try:

#     print(f"Processing player {player['id']}")#test

#     # extract relevant data for each player
#     player_id = player['id']
#     player_name = player['web_name']
#     team_id = player['team']
#     position_id = player['element_type']

#     # make a request to second API to get detailed player info
#     print(f"Fetching detailed info for player {player_id}")
#     response = requests.get(f'https://fantasy.premierleague.com/api/element-summary/{player_id}/')

#     # if response code is not 200, raise an exception
#     if response.status_code != 200:
#       raise Exception(f"Unexpected status code {response.status_code}. Message: {response.text}")

#     detailed_info = response.json()
#     history = detailed_info['history']


#     for game in history:
#       print(f"Processing game {game['round']} for player {player_id}") # debug print

#       round_number = game['round']
#       goals_scored = game['goals_scored']
#       assists = game['assists']
#       total_points = game['total_points']
#       expected_goal_involvements = game['expected_goal_involvements']
#       ict_index = game['ict_index']

#       # insert player's data into the Performance table
#       insert_performance = """
#       INSERT OR REPLACE INTO gameweeks (id, round, expected_goal_involvements, ict_index, goals_scored, assists)
#       VALUES (?, ?, ?, ?, ?, ?);
#       """
#       cur.execute(insert_performance, (player_id, round_number, expected_goal_involvements, ict_index, goals_scored, assists))

#       # print message
#       print(f"Data to be inserted: {player_id, round_number, expected_goal_involvements, ict_index, goals_scored, assists}")

#   except Exception as e:
#     print(f"An error occurred when processing player {player_id}: {e}")
#     continue

# # commit changes
# conn.commit()

# # close connection
# conn.close()









## test data retrieval
# # execute simple SQL command
# cur.execute("SELECT * FROM players")

# # fetch all rows - return a list of tuples where each tuple represents a row
# players = cur.fetchall()

# #print players
# for player in players:
#     print(player)

# # close connection
# conn.close()



# # test data insertion
# player_data = (1, 'Pele', 'Forward', 'Santos')

# # insert row of data
# cur.execute("INSERT INTO players VALUES (?, ?, ?, ?)", player_data)

# # commit changes
# conn.commit()

# plots & histograms
# show first few ros of data
# print(df.head())

# plot CFi vs next game goals
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='CFi', y='next_game_goals', data=df)
# plt.savefig('CFi_vs_next_game_goals.png')

# #plot ict_index vs next game goals
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='ict_index', y='next_game_goals', data=df)
# plt.savefig('ict_index_vs_next_game_goals.png')

# # plot expected_goal_involvements vs next game goals
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='expected_goal_involvements', y='next_game_goals', data=df)
# plt.savefig('expected_goal_involvements_vs_next_game_goals.png')

# print statistical summary
#correlation heatmap
# numeric_columns = df.select_dtypes(include=['int64', 'float64'])
# correlation = numeric_columns.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation, annot=True, cmap='coolwarm')
# plt.savefig('heatmap.png')

# linear regression
# # create linear regression model
# model = LinearRegression()

# # use CFi as the feature to get next game goals and assists as target
# features = df[['CFi']]
# target = df[['next_game_goals', 'next_game_assists']]

# # perform cross-validation
# scores = cross_val_score(model, features, target, cv=5)

# #print scores
# print(scores)


# # define independent variable (X) and dependent variable (y)
# X_train = df[df['round'] <= 16][['CFi']]
# y_train = df[df['round'] <= 16][['next_game_goals', 'next_game_assists']]

# X_test = df[df['round'] >= 21][['CFi']]
# y_test = df[df['round'] >= 21][['next_game_goals', 'next_game_assists']]

# # train the model
# model.fit(X_train, y_train)

# #make predictions on test set
# predictions = model.predict(X_test)

# #print the first 10 predictions
# print(predictions[:10])

# #print first 10 actual outcomes
# print(y_test[:10])

# #calculate MSE
# mse = mean_squared_error(y_test, predictions)

# #print MSE
# print(mse)

#Random Forest testing
# Random Forest testing Day 1
# #query the needed SQL data
# query = """
# SELECT round, expected_goal_involvements, ict_index, goals_scored, assists, player_name, team, position, CFi,
#         CASE
#             WHEN next_game_goals > 0 OR next_game_assists > 0 THEN 1
#             ELSE 0
#         END AS scored_or_assisted_next_game
# FROM gameweeks
# WHERE round BETWEEN 5 AND 16 OR round BETWEEN 21 AND 36
# """

# # use pd.read_sql_query to query the database and put the result in a dataframe
# df = pd.read_sql_query(query, conn)


# #close connection to database
# conn.close()

# #define the features and target
# X = df[['CFi', 'ict_index', 'expected_goal_involvements']]
# y= df['scored_or_assisted_next_game']

# #split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df[['CFi']], df['scored_or_assisted_next_game'], test_size=0.3, random_state=42)

# #define the Random Forest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)

# # fit the model to the training data
# model.fit(X_train, y_train)

# #make preditions on the testing data
# y_pred = model.predict(X_test)

# #print the model's accuracy
# print("Accuracy: ", accuracy_score(y_test, y_pred))

# #print the confusion matrix
# print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# #print the classification report
# print("Classification Report \n", classification_report(y_test, y_pred))

# # print feature importances
# feature_importances = pd.DataFrame(model.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
# print(feature_importances)
# ##training for binary target variable using Logistic Regression


# #initialize logistic regression classifier
# logreg = LogisticRegression()

# # train the classifier
# logreg.fit(X_train, y_train)

# #makes predictions on the test set
# y_pred = logreg.predict(X_test)

# #print the accuracy score
# print("Accuracy: ", accuracy_score(y_test, y_pred))

# #print the confusion matrix
# print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# #print the classification report
# print("Classification Report \n", classification_report(y_test, y_pred))

# ##TUNE RANDOM FOREST MODEL
# engine = create_engine('sqlite:///fantasy_football.db')

# # SQL query to fetch data
# query = """
# SELECT *
# FROM gameweeks
# """

# #fetch data using pandas
# df = pd.read_sql_query(query, engine)

# #preprocess data
# #convery 'next_game_goals' and 'next_game_assists' into binary target variable
# df['target'] = ((df['next_game_goals'] > 0) | (df['next_game_assists'] > 0)).astype(int)

# # specify the features and the target
# features = ['round', 'expected_goal_involvements', 'ict_index', 'goals_scored', 'assists', 'CFi']
# target = 'target'

# #split data into features (X) and target (y)
# X = df[features]
# y = df[target]

# #split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_leaf':[1, 2, 4],
#     'min_samples_split': [2, 5, 10]
# }

# #create base model to tune
# rf = RandomForestClassifier()

# #instantiate the grid search model
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# # fit the grid to search the data
# grid_search.fit(X_train, y_train)

# #print the best parameters
# print(grid_search.best_params_)

# #print the best score
# print(grid_search.best_score_)


# ## REVISE CFi VALUES
# #connect to SQL database
# engine = create_engine('sqlite:///fantasy_football.db')

# #pull data from database into pandas Dataframe
# query = """
#     SELECT round, expected_goal_involvements, ict_index, goals_scored, assists, player_name
#     FROM gameweeks
# """

# df = pd.read_sql_query(query, engine)

# #ensure data is sorted by round, then player_name
# df = df.sort_values(by=['player_name', 'round'])

# #create columns for past expected_goal_involvements, goals_scored, and assists
# df['past_xG1'] = df.groupby('player_name')['expected_goal_involvements'].shift(1)
# df['past_xG2'] = df.groupby('player_name')['expected_goal_involvements'].shift(2)
# df['past_xG3'] = df.groupby('player_name')['expected_goal_involvements'].shift(3)
# df['past_Goals1'] = df.groupby('player_name')['goals_scored'].shift(1)
# df['past_Goals2'] = df.groupby('player_name')['goals_scored'].shift(2)
# df['past_Goals3'] = df.groupby('player_name')['goals_scored'].shift(3)
# df['past_Assists1'] = df.groupby('player_name')['assists'].shift(1)
# df['past_Assists2'] = df.groupby('player_name')['assists'].shift(2)
# df['past_Assists3'] = df.groupby('player_name')['assists'].shift(3)

# # compute new CFi
# df['CFi'] = np.where(
#     (df['past_Goals1'] > 0) | (df['past_Goals2'] > 0) | (df['past_Goals3'] > 0) |
#     (df['past_Assists1'] > 0) | (df['past_Assists2'] > 0) | (df['past_Assists3'] > 0),
#     0,
#     (df['past_xG3'] * 0.8) + (df['past_xG2'] * 0.9) + (df['past_xG1']) + (df['ict_index'] * 0.001)
# )
# #replace NaN values with 0
# df['CFi'] = df['CFi'].fillna(0)

# # drop the past_xG, past_Goals, and past_Assists columns as they are no longer needed
# df.drop(['past_xG1', 'past_xG2', 'past_xG3', 'past_Goals1', 'past_Goals2', 'past_Goals3', 'past_Assists1', 'past_Assists2', 'past_Assists3'], axis=1, inplace=True)

# # update CFi values with new SQL data
# with engine.connect() as conn:
#   for i, row in df.iterrows():
#     player_name = text(conn.dialect.identifier_preparer.quote(row['player_name']))
#     conn.execute(f"""
#         UPDATE gameweeks
#         SET CFi = {row['CFi']}
#         WHERE round = {row['round']} AND player_name = {player_name}
#     """)

# #load data from SQL to dataframe
# # connec to SQL
# conn = sqlite3.connect('fantasy_football.db')

# #TUNE RANDOM FOREST THRESHOLD DOWNWARD
# # read from SQL
# engine = create_engine('sqlite:///fantasy_football.db')
# data = pd.read_sql("SELECT * FROM gameweeks", engine)

# # Fill NaN values with 0
# data['next_game_goals'].fillna(0, inplace=True)
# data['next_game_assists'].fillna(0, inplace=True)

# #set target and features
# y = ((data['next_game_goals'] > 0) | (data['next_game_assists'] > 0)).astype(int)
# X = data[['CFi']]

# # train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # instantiate and fit Random Forest classifier with optimal hyperparameters
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# rfc = RandomForestClassifier()
# grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, verbose=2)
# grid_search.fit(X_train, y_train)

# #get best parameters and score from Grid Search
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# #use the fitted model to predict the training instances
# y_train_pred_proba = grid_search.predict_proba(X_train)
# y_test_pred_proba = grid_search.predict_proba(X_test)

# # apply the threshold
# threshold = 0.3
# y_train_pred = (y_train_pred_proba[:,1] >= threshold).astype(int)
# y_test_pred = (y_test_pred_proba[:,1] >= threshold).astype(int)

# #print the confusion matrix
# cm = confusion_matrix(y_test, y_test_pred)
# print("Confusion Matrix: ")
# print(cm)

# #print the Classification Report
# cr = classification_report(y_test, y_test_pred)
# print("Classification Report: ")
# print(cr)

# #APPLY TUNED RANDOM FOREST MODEL TO DATA
# engine = create_engine('sqlite:///fantasy_football.db')
# df = pd.read_sql_query("SELECT * FROM gameweeks", engine)

# # fill in missing values with zeros
# df = df.fillna(0)

# #create binary target column
# df['scored_or_assisted'] = ((df['next_game_goals'] > 0) | (df['next_game_assists'] > 0)).astype(int)

# #split data into training and test set
# X = df[['CFi']]
# y = df['scored_or_assisted']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #inititalize Random Forest Classifier
# rf = RandomForestClassifier()

# #specify hyperparameters and values to test in grid search
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# #intialize a GridSearchCV object that will find the best hyperparameters
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2)

# # fit GridSearchCV object to the data
# grid_search.fit(X_train, y_train)

# #print best parameters and accuracy score
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# #initialize model with best parameters
# rf_best = RandomForestClassifier(
#     max_depth=grid_search.best_params_['max_depth'],
#     min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
#     min_samples_split=grid_search.best_params_['min_samples_split'],
#     n_estimators=grid_search.best_params_['n_estimators']
# )

# # fit the model on the training data
# rf_best.fit(X_train, y_train)

# #predict the target for the test data
# y_pred = rf_best.predict(X_test)

# #calculate accuracy on test set
# accuracy = accuracy_score(y_test, y_pred)

# #print accuracy
# print(f"Accuracy on the test set: {accuracy}")

# #print the confusion matrix
# print(confusion_matrix(y_test, y_pred))

# # print the classification report
# print(classification_report(y_test, y_pred))

#BALANCE DATASET USING SMOTE
# #load data from SQL database
# engine = create_engine('sqlite:///fantasy_football.db')
# data = pd.read_sql("SELECT * FROM gameweeks", engine)


# #convert team and position to dummy variables
# data = pd.get_dummies(data, columns=['team', 'position'])

# # set up our X and y
# X = data[['CFi', 'ict_index'] + [col for col in data.columns if 'team_' in col or 'position_' in col]]
# y = ((data['next_game_goals'] > 0) | (data['next_game_assists'] > 0)).astype(int)

# #split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #apply SMOTE
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # set up Random Forest Classifier
# clf = RandomForestClassifier(random_state=42)

# #use GridSearch to find best parameters
# params = {'n_estimators': [50, 100, 200], 'max_depth': [100, 20, 30, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
# grid_search = GridSearchCV(clf, param_grid=params, cv=3, verbose=2)
# grid_search.fit(X_train_smote, y_train_smote)

# #predict the test set results
# y_pred = grid_search.predict(X_test)

# #save model to disk/joblib file
# dump(clf, 'fantasy_football_model.joblib')





























# #BALANCE DATASET USING SMOTE
# #load data from SQL database
# engine = create_engine('sqlite:///fantasy_football.db')
# data = pd.read_sql("SELECT * FROM gameweeks", engine)


# #convert team and position to dummy variables
# data = pd.get_dummies(data, columns=['team', 'position'])

# # set up our X and y
# X = data[['CFi', 'ict_index'] + [col for col in data.columns if 'team_' in col or 'position_' in col]]
# y = ((data['next_game_goals'] > 0) | (data['next_game_assists'] > 0)).astype(int)

# #split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #apply SMOTE
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # set up Random Forest Classifier
# clf = RandomForestClassifier(random_state=42)

# #use GridSearch to find best parameters
# params = {'n_estimators': [50, 100, 200], 'max_depth': [100, 20, 30, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
# grid_search = GridSearchCV(clf, param_grid=params, cv=3, verbose=2)
# grid_search.fit(X_train_smote, y_train_smote)

# #predict the test set results
# y_pred = grid_search.predict(X_test)

# #save model to disk/joblib file
# dump(clf, 'fantasy_football_model.joblib')

# # evaluate the model
# print(grid_search.best_params_)
# print(grid_search.best_score_)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

#convert
# #TUNE RANDOM FOREST THRESHOLD DOWNWARD
# # read from SQL
# engine = create_engine('sqlite:///fantasy_football.db')
# data = pd.read_sql("SELECT * FROM gameweeks", engine)

# # Fill NaN values with 0
# data['next_game_goals'].fillna(0, inplace=True)
# data['next_game_assists'].fillna(0, inplace=True)

# #set target and features
# y = ((data['next_game_goals'] > 0) | (data['next_game_assists'] > 0)).astype(int)
# X = data[['CFi']]

# # train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # instantiate and fit Random Forest classifier with optimal hyperparameters
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# rfc = RandomForestClassifier()
# grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, verbose=2)
# grid_search.fit(X_train, y_train)

# #get best parameters and score from Grid Search
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# #use the fitted model to predict the training instances
# y_train_pred_proba = grid_search.predict_proba(X_train)
# y_test_pred_proba = grid_search.predict_proba(X_test)

# # apply the threshold
# threshold = 0.3
# y_train_pred = (y_train_pred_proba[:,1] >= threshold).astype(int)
# y_test_pred = (y_test_pred_proba[:,1] >= threshold).astype(int)

# #print the confusion matrix
# cm = confusion_matrix(y_test, y_test_pred)
# print("Confusion Matrix: ")
# print(cm)

# #print the Classification Report
# cr = classification_report(y_test, y_test_pred)
# print("Classification Report: ")
# print(cr)

# #APPLY TUNED RANDOM FOREST MODEL TO DATA
# engine = create_engine('sqlite:///fantasy_football.db')
# df = pd.read_sql_query("SELECT * FROM gameweeks", engine)

# # fill in missing values with zeros
# df = df.fillna(0)

# #create binary target column
# df['scored_or_assisted'] = ((df['next_game_goals'] > 0) | (df['next_game_assists'] > 0)).astype(int)

# #split data into training and test set
# X = df[['CFi']]
# y = df['scored_or_assisted']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #inititalize Random Forest Classifier
# rf = RandomForestClassifier()

# #specify hyperparameters and values to test in grid search
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# #intialize a GridSearchCV object that will find the best hyperparameters
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2)

# # fit GridSearchCV object to the data
# grid_search.fit(X_train, y_train)

# #print best parameters and accuracy score
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# #initialize model with best parameters
# rf_best = RandomForestClassifier(
#     max_depth=grid_search.best_params_['max_depth'],
#     min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
#     min_samples_split=grid_search.best_params_['min_samples_split'],
#     n_estimators=grid_search.best_params_['n_estimators']
# )

# # fit the model on the training data
# rf_best.fit(X_train, y_train)

# #predict the target for the test data
# y_pred = rf_best.predict(X_test)

# #calculate accuracy on test set
# accuracy = accuracy_score(y_test, y_pred)

# #print accuracy
# print(f"Accuracy on the test set: {accuracy}")

# #print the confusion matrix
# print(confusion_matrix(y_test, y_pred))

# # print the classification report
# print(classification_report(y_test, y_pred))


# ##TUNE RANDOM FOREST MODEL
# engine = create_engine('sqlite:///fantasy_football.db')

# # SQL query to fetch data
# query = """
# SELECT *
# FROM gameweeks
# """

# #fetch data using pandas
# df = pd.read_sql_query(query, engine)

# #preprocess data
# #convery 'next_game_goals' and 'next_game_assists' into binary target variable
# df['target'] = ((df['next_game_goals'] > 0) | (df['next_game_assists'] > 0)).astype(int)

# # specify the features and the target
# features = ['round', 'expected_goal_involvements', 'ict_index', 'goals_scored', 'assists', 'CFi']
# target = 'target'

# #split data into features (X) and target (y)
# X = df[features]
# y = df[target]

# #split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_leaf':[1, 2, 4],
#     'min_samples_split': [2, 5, 10]
# }

# #create base model to tune
# rf = RandomForestClassifier()

# #instantiate the grid search model
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# # fit the grid to search the data
# grid_search.fit(X_train, y_train)

# #print the best parameters
# print(grid_search.best_params_)

# #print the best score
# print(grid_search.best_score_)


# ## REVISE CFi VALUES
# #connect to SQL database
# engine = create_engine('sqlite:///fantasy_football.db')

# #pull data from database into pandas Dataframe
# query = """
#     SELECT round, expected_goal_involvements, ict_index, goals_scored, assists, player_name
#     FROM gameweeks
# """

# df = pd.read_sql_query(query, engine)

# #ensure data is sorted by round, then player_name
# df = df.sort_values(by=['player_name', 'round'])

# #create columns for past expected_goal_involvements, goals_scored, and assists
# df['past_xG1'] = df.groupby('player_name')['expected_goal_involvements'].shift(1)
# df['past_xG2'] = df.groupby('player_name')['expected_goal_involvements'].shift(2)
# df['past_xG3'] = df.groupby('player_name')['expected_goal_involvements'].shift(3)
# df['past_Goals1'] = df.groupby('player_name')['goals_scored'].shift(1)
# df['past_Goals2'] = df.groupby('player_name')['goals_scored'].shift(2)
# df['past_Goals3'] = df.groupby('player_name')['goals_scored'].shift(3)
# df['past_Assists1'] = df.groupby('player_name')['assists'].shift(1)
# df['past_Assists2'] = df.groupby('player_name')['assists'].shift(2)
# df['past_Assists3'] = df.groupby('player_name')['assists'].shift(3)

# # compute new CFi
# df['CFi'] = np.where(
#     (df['past_Goals1'] > 0) | (df['past_Goals2'] > 0) | (df['past_Goals3'] > 0) |
#     (df['past_Assists1'] > 0) | (df['past_Assists2'] > 0) | (df['past_Assists3'] > 0),
#     0,
#     (df['past_xG3'] * 0.8) + (df['past_xG2'] * 0.9) + (df['past_xG1']) + (df['ict_index'] * 0.001)
# )
# #replace NaN values with 0
# df['CFi'] = df['CFi'].fillna(0)

# # drop the past_xG, past_Goals, and past_Assists columns as they are no longer needed
# df.drop(['past_xG1', 'past_xG2', 'past_xG3', 'past_Goals1', 'past_Goals2', 'past_Goals3', 'past_Assists1', 'past_Assists2', 'past_Assists3'], axis=1, inplace=True)

# # update CFi values with new SQL data
# with engine.connect() as conn:
#   for i, row in df.iterrows():
#     player_name = text(conn.dialect.identifier_preparer.quote(row['player_name']))
#     conn.execute(f"""
#         UPDATE gameweeks
#         SET CFi = {row['CFi']}
#         WHERE round = {row['round']} AND player_name = {player_name}
#     """)

# #load data from SQL to dataframe
# # connec to SQL
# conn = sqlite3.connect('fantasy_football.db')





# Random Forest testing Day 1
# #query the needed SQL data
# query = """
# SELECT round, expected_goal_involvements, ict_index, goals_scored, assists, player_name, team, position, CFi,
#         CASE
#             WHEN next_game_goals > 0 OR next_game_assists > 0 THEN 1
#             ELSE 0
#         END AS scored_or_assisted_next_game
# FROM gameweeks
# WHERE round BETWEEN 5 AND 16 OR round BETWEEN 21 AND 36
# """

# # use pd.read_sql_query to query the database and put the result in a dataframe
# df = pd.read_sql_query(query, conn)


# #close connection to database
# conn.close()

# #define the features and target
# X = df[['CFi', 'ict_index', 'expected_goal_involvements']]
# y= df['scored_or_assisted_next_game']

# #split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df[['CFi']], df['scored_or_assisted_next_game'], test_size=0.3, random_state=42)

# #define the Random Forest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)

# # fit the model to the training data
# model.fit(X_train, y_train)

# #make preditions on the testing data
# y_pred = model.predict(X_test)

# #print the model's accuracy
# print("Accuracy: ", accuracy_score(y_test, y_pred))

# #print the confusion matrix
# print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# #print the classification report
# print("Classification Report \n", classification_report(y_test, y_pred))

# # print feature importances
# feature_importances = pd.DataFrame(model.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
# print(feature_importances)
# ##training for binary target variable using Logistic Regression


# #initialize logistic regression classifier
# logreg = LogisticRegression()

# # train the classifier
# logreg.fit(X_train, y_train)

# #makes predictions on the test set
# y_pred = logreg.predict(X_test)

# #print the accuracy score
# print("Accuracy: ", accuracy_score(y_test, y_pred))

# #print the confusion matrix
# print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# #print the classification report
# print("Classification Report \n", classification_report(y_test, y_pred))



# # create linear regression model
# model = LinearRegression()

# # use CFi as the feature to get next game goals and assists as target
# features = df[['CFi']]
# target = df[['next_game_goals', 'next_game_assists']]

# # perform cross-validation
# scores = cross_val_score(model, features, target, cv=5)

# #print scores
# print(scores)


# # define independent variable (X) and dependent variable (y)
# X_train = df[df['round'] <= 16][['CFi']]
# y_train = df[df['round'] <= 16][['next_game_goals', 'next_game_assists']]

# X_test = df[df['round'] >= 21][['CFi']]
# y_test = df[df['round'] >= 21][['next_game_goals', 'next_game_assists']]

# # train the model
# model.fit(X_train, y_train)

# #make predictions on test set
# predictions = model.predict(X_test)

# #print the first 10 predictions
# print(predictions[:10])

# #print first 10 actual outcomes
# print(y_test[:10])

# #calculate MSE
# mse = mean_squared_error(y_test, predictions)

# #print MSE
# print(mse)






# show first few ros of data
# print(df.head())

# plot CFi vs next game goals
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='CFi', y='next_game_goals', data=df)
# plt.savefig('CFi_vs_next_game_goals.png')

# #plot ict_index vs next game goals
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='ict_index', y='next_game_goals', data=df)
# plt.savefig('ict_index_vs_next_game_goals.png')

# # plot expected_goal_involvements vs next game goals
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='expected_goal_involvements', y='next_game_goals', data=df)
# plt.savefig('expected_goal_involvements_vs_next_game_goals.png')

# print statistical summary
#correlation heatmap
# numeric_columns = df.select_dtypes(include=['int64', 'float64'])
# correlation = numeric_columns.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation, annot=True, cmap='coolwarm')
# plt.savefig('heatmap.png')





# ## to load data from pickle to SQL
# #load data
# data = pd.read_pickle('player_data.pkl')

# # check if data is a dictionary
# if isinstance(data, dict):
#   # convert dict of Dataframes to one DataFrame
#   df = pd.concat(data.values(), ignore_index=True)
# else:
#   df = data

# conn = sqlite3.connect('fantasy_football.db')

# cursor = conn.cursor()
# cursor.execute('DROP TABLE IF EXISTS gameweeks')
# conn.commit()

# df.to_sql('gameweeks', conn, if_exists='replace', index=False)

# conn.close()

# # # check if data is list of DataFrames
# if isinstance(data, dict):
#     # assign each DataFrame with new column equal to dictionary key
#     for key in data:
#         data[key]['key'] = key
#     #concatenate all DataFrames in list
#     df = pd.concat(data.values(), ignore_index=True)
# elif isinstance(data, list):
#     df = pd.concat(data, ignore_index=True)
# else:
#     # if data is not a list, its already a DataFrame
#     df = data

# print(df)

# # establish connection
# conn = sqlite3.connect('fantasy_football.db')

# # save DataFrame to SQL
# df.to_sql('gameweeks', conn, if_exists='append', index=False)

# conn.close()

# # write the data
# df.to_sql('gameweeks', conn, if_exists='append', index=False)

# # create a cursor object to execute SQL commands
# cur = conn.cursor()

# # add additional fields into SQL
# cur.execute("ALTER TABLE gameweeks ADD COLUMN key INTEGER")
# # cur.execute("ALTER TABLE gameweeks ADD COLUMN position INTEGER")

# conn.commit()



# with open('player_data.pkl', 'rb') as f:
#     data = pickle.load(f)





# # connection to SQLite
# conn = sqlite3.connect('fantasy_football.db')

# # loop over data dict
# for player_id, df in data.items():
#     # add player_id column to the DataFrame
#     df['player_id'] = player_id
#     # write the DataFrame to SQL database
#     df.to_sql('gameweeks', conn, if_exists='append', index=False)

# conn.close()




# # check data type
# print(type(data))

# # if its a list or dict, check length or number of keys
# if isinstance(data, list):
#     print(len(data))
# elif isinstance(data, dict):
#     print(len(data.keys()))







# SQL DATABASE
# # establish connection to SQLite database
# conn = sqlite3.connect('fantasy_football.db')


# # create a cursor object to execute SQL commands
# cur = conn.cursor()

# # delete all rows from the gameweeks table
# cur.execute("DELETE FROM gameweeks")

# # set API endpoint
# url = "https://fantasy.premierleague.com/api/bootstrap-static/"

# # send GET response to API
# response = requests.get(url)

# # parse JSON response
# data = json.loads(response.text)

# # extract players data
# players_data = data['elements']

# # loop over players data
# for player in players_data:

#   try:

#     print(f"Processing player {player['id']}")#test

#     # extract relevant data for each player
#     player_id = player['id']
#     player_name = player['web_name']
#     team_id = player['team']
#     position_id = player['element_type']

#     # make a request to second API to get detailed player info
#     print(f"Fetching detailed info for player {player_id}")
#     response = requests.get(f'https://fantasy.premierleague.com/api/element-summary/{player_id}/')

#     # if response code is not 200, raise an exception
#     if response.status_code != 200:
#       raise Exception(f"Unexpected status code {response.status_code}. Message: {response.text}")

#     detailed_info = response.json()
#     history = detailed_info['history']


#     for game in history:
#       print(f"Processing game {game['round']} for player {player_id}") # debug print

#       round_number = game['round']
#       goals_scored = game['goals_scored']
#       assists = game['assists']
#       total_points = game['total_points']
#       expected_goal_involvements = game['expected_goal_involvements']
#       ict_index = game['ict_index']

#       # insert player's data into the Performance table
#       insert_performance = """
#       INSERT OR REPLACE INTO gameweeks (id, round, expected_goal_involvements, ict_index, goals_scored, assists)
#       VALUES (?, ?, ?, ?, ?, ?);
#       """
#       cur.execute(insert_performance, (player_id, round_number, expected_goal_involvements, ict_index, goals_scored, assists))

#       # print message
#       print(f"Data to be inserted: {player_id, round_number, expected_goal_involvements, ict_index, goals_scored, assists}")

#   except Exception as e:
#     print(f"An error occurred when processing player {player_id}: {e}")
#     continue

# # commit changes
# conn.commit()

# # close connection
# conn.close()









## test data retrieval
# # execute simple SQL command
# cur.execute("SELECT * FROM players")

# # fetch all rows - return a list of tuples where each tuple represents a row
# players = cur.fetchall()

# #print players
# for player in players:
#     print(player)

# # close connection
# conn.close()



# # test data insertion
# player_data = (1, 'Pele', 'Forward', 'Santos')

# # insert row of data
# cur.execute("INSERT INTO players VALUES (?, ?, ?, ?)", player_data)

# # commit changes
# conn.commit()















#   # create a Gameweeks table
#   cur.execute("""
#       CREATE TABLE gameweeks (
#         id INTEGER NOT NULL,
#         round INTEGER NOT NULL,
#         expected_goal_involvements REAL,
#         ict_index REAL,
#         goals_scored INTEGER,
#         assists INTEGER,
#         CFi REAL,
#         PRIMARY KEY (id, round),
#         FOREIGN KEY (id) REFERENCES players (id)
#       )
#   """)

#   print("Gameweeks table created successfully.")

# except Exception as e:
#   print(f"An error occurred: {e}")

# finally:
#   # close connection after operations done
#   conn.close()

#   # create the Players table
#   cur.execute("""
#       CREATE TABLE players (
#         id INTEGER PRIMARY KEY,
#         name TEXT NOT NULL,
#         team TEXT NOT NULL,
#         position INTEGER NOT NULL
#       )
#   """)

#   print("Players table created successfully.")



# except Exception as e:
#   print(f"An error occurred: {e}")

# finally:
#   # close connection after operations are done
#   conn.close()
































# # Fetch all player data
# url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
# response = requests.get(url)
# data = response.json()

# # extract the player information from the response
# players = data['elements']

# # prepare a dictionary to hold our data
# player_data = {}

# # loop over the players
# for player in players:
#     #skip goalkeepers
#     if player['element_type'] == 1:
#         continue
#     # for the remaining players, fetch their detailed info
#     response = requests.get(f'https://fantasy.premierleague.com/api/element-summary/{player["id"]}/')
#     player_info = response.json()

#     # extract the info for each round and add it to the dictionary
#     for round_info in player_info['history']:
#         round_number = round_info['round']
#         player_data.setdefault(player['id'], {}).setdefault(round_number, {
#             'round': round_number,
#             'expected_goal_involvements': round_info['expected_goal_involvements'],
#             'ict_index': round_info['ict_index'],
#             'goals_scored': round_info['goals_scored'],
#             'assists': round_info['assists'],
#             'player_name': player['web_name'],
#             'team': player['team'],
#             'position': player['element_type']
#         })
#     #at this point, 'player_data' is a dictionary where the key is the player id and the value is a DataFrame containing player's info and historical data

# # save the data
# with open('player_data.pkl', 'wb') as f:
#     pickle.dump(player_data, f)

#load the data
# with open('player_data.pkl', 'rb') as f:
#     player_data = pickle.load(f)

# for player_id, player_dict in player_data.items():
#   player_df = pd.DataFrame(player_dict) # convert dictionary to DataFrame

#   print(player_df.columns)
#   print(player_df.head())

#   # Transpose the dataframe
#   player_df = player_df.transpose()

#   # convert round from index to column
#   player_df.reset_index(level=0, inplace=True)
#   player_df.rename(columns={'index':'round'}, inplace=True)

#   # convert round to numeric
#   player_df['round'] = pd.to_numeric(player_df['round'], errors='coerce')

#   # get expected goal involvements (xG) for the previous weeks
#   player_df["gW4_xG"] = player_df["expected_goal_involvements"].shift(4).fillna(0)
#   player_df["gW3_xG"] = player_df["expected_goal_involvements"].shift(3).fillna(0)
#   player_df["gW2_xG"] = player_df["expected_goal_involvements"].shift(2).fillna(0)
#   player_df["gW1_xG"] = player_df["expected_goal_involvements"].shift(1).fillna(0)

#   # compute CFi
#   player_df["CFi"] = (player_df["gW4_xG"] * 0.7) + (player_df["gW3_xG"] * 0.8) + \
#                    (player_df["gW2_xG"] * 0.9) + player_df["gW1_xG"] + (player_df["ict_index"] * 0.001)

#   # remove temporary xG columns

#   player_df.drop(columns=["gW4_xG", "gW3_xG", "gW2_xG", "gW1_xG"], inplace=True)

#   # add in next game goals and assist columns
#   player_df["next_game_goals"] = player_df["goals_scored"].shift(-1)
#   player_df["next_game_assists"] = player_df["assists"].shift(-1)

#   # drop the last row
#   player_df = player_df.iloc[:-1]

#   # update player data in dictionary
#   player_data[player_id] = player_df

# print(player_df.head())


# # print the data for the first 3 players
# for i, (player_id, player_dict) in enumerate(player_data.items()):
#     if i > 2: # only print for first 3 players
#         break

#     print(f"Player ID: {player_id}")
#     player_df = pd.DataFrame(player_dict).transpose() #convert dict to DataFrame
#     print(player_df.head())

# #compute CFi for each player for each round
# for player_id, player_dict in player_data.items():
#     player_df = pd.DataFrame(player_dict) #convert dictionary to DataFrame



#     #convert data types
#     numeric_columns = ["expected_goal_involvements", "ict_index", "goals_scored", "assists"]
#     for column in numeric_columns:
#         if column in player_df.columns:
#             player_df[column] = player_df[column].astype(float)
#             player_df[column] = pd.to_numeric(player_df[column], errors='coerce')








#     # debug lines
#     # print(f"Player ID: {player_id}")
#     # print(player_df.dtypes)
#     # print(player_df.head())

#     # get expected goals involvements (xG) for the previous weeks
#     player_df["gW4_xG"] = player_df["expected_goal_involvements"].shift(4).fillna(0)
#     player_df["gW3_xG"] = player_df["expected_goal_involvements"].shift(3).fillna(0)
#     player_df["gW2_xG"] = player_df["expected_goal_involvements"].shift(2).fillna(0)
#     player_df["gW1_xG"] = player_df["expected_goal_involvements"].shift(1).fillna(0)

#     #compute CFi
#     player_df["CFi"] = (player_df["gW4_xG"] * 0.7) + (player_df["gW3_xG"] * 0.8) + \
#                         (player_df["gW2_xG"] * 0.9) + player_df["gW1_xG"] + (player_df["ict_index"] * 0.001)

#     # remove temporary xG columns
#     player_df.drop(columns=["gW4_xG", "gW3_xG", "gW2_xG", "gW1_xG"], inplace=True)

#     # add in next game goals and assist columns
#     player_df["next_game_goals"] = player_df["goals_scored"].shift(-1)
#     player_df["next_game_assists"] = player_df["assists"].shift(-1)

#     # drop the last row
#     player_df = player_df.iloc[:-1]

#     # update player data in the dictionary
#     player_data[player_id] = player_df

# # save updated data back to pickle file
# with open("player_data.pkl", "wb") as f:
#     pickle.dump(player_data, f)

# # load the data
# with open('player_data.pkl', 'rb') as f:
#     player_data = pickle.load(f)

# # print the data for the first 3 players
# for i, (player_id, player_dict) in enumerate(player_data.items()):
#     if i > 2: # only print for first 3 players
#         break

#     print(f"Player ID: {player_id}")
#     player_df = pd.DataFrame(player_dict).transpose() #convert dict to DataFrame
#     print(player_df.head())



# converting dictionary to dataframe and then to CSV
# all_players_data = []

# for player_id, player_info in player_data.items():
#     for round, round_info in player_info.items():
#         round_info['player_id'] = player_id
#         round_info['round'] = round
#         all_players_data.append(round_info)

# df = pd.DataFrame(all_players_data)
# df.to_csv('player_data.csv', index=False)






























#pd.options.display.max_columns=None

# def get_data(url):
#     response = requests.get(url)
#     #json = response.json()
#     #df = pd.DataFrame(json)
#     return response.json()


# def fetch_player_history(player_id):
#     # API URL
#     player_url = f'https://fantasy.premierleague.com/api/element-summary/{player_id}/'

#     # get player data
#     player_data = get_data(player_url)

#     #extract historical data
#     history_df = pd.DataFrame(player_data['history'])

#     # filter out unnecessary columns
#     history_df = history_df[['round', 'expected_goal_involvements', 'ict_index', 'goals_scored', 'assists']]

#     return history_df

## initial fetch from API
# # get elements df
# url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
# data = get_data(url)
# elements_df = pd.DataFrame(data['elements'])

# #get list of player IDs
# player_ids = elements_df['id'].values

# #initialize dictionary to hold player data
# player_data = {}

# #fetch data for players
# #for player_id in player_ids:
#  #   print(f"Fetching data for player {player_id}...")
#   #  player_data[player_id] = fetch_player_history(player_id)

# #with open('player_data.pickle_2', 'wb') as handle:
#  #   pickle.dump(player_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# with open('player_data_2.pickle', 'rb') as handle:
#     player_data = pickle.load(handle)


#CFI calculate function
#def calculate_cfi(player_data):
  #  for player_id, history_data in player_data.items():
   #     df = pd.DataFrame(history_data)
    #    df = df[['round', 'expected_goal_involvements', 'ict_index', 'goals_scored', 'assists']]

        #order the dataframe by gameweek, most recent game last
     #   df.sort_values('round', inplace=True)

        #check if dataframe is empty or if the player has scored or assisted in the last four gameweeks
      #  if df.empty or df.iloc[-4:][['goals_scored', 'assists']].sum().sum() > 0 or len(df) < 4:
       #     player_data[player_id]['CFi'] = 0
        #else:
         #   gW1_xG = df.iloc[-1]['expected_goal_involvements']
          #  gW2_xG = df.iloc[-2]['expected_goal_involvements']
           # gW3_xG = df.iloc[-3]['expected_goal_involvements']
            #gW4_xG = df.iloc[-4]['expected_goal_involvements']
            #iCT = df.iloc[-1]['ict_index']

            #CFi = (gW4_xG * 0.7) + (gW3_xG * 0.8) + (gW2_xG * 0.9) + gW1_xG + (iCT * 0.001)
            #player_data[player_id]['CFi'] = CFi

    #return player_data

# def calculate_cfi(player_df):
#     # if player_df is a dictionary, convert to dataframe
#     if isinstance(player_df, dict):
#         df = pd.DataFrame(player_df).T
#     else:
#         df = player_df.copy()

#     #print columns for debugging
#     #print(df.columns)

#     df = df[['round', 'expected_goal_involvements', 'ict_index', 'goals_scored', 'assists']]

#     #skip if not enough data
#     if df.empty or df['goals_scored'].sum() > 0 or df['assists'].sum() > 0 or len(df) < 4:
#         return None

#     #calculate CFi
#     df['xG'] = df['expected_goal_involvements'].astype(float)
#     df['iCT'] = df['ict_index'].astype(float)
#     gW4_xG = df.iloc[-4]['xG'] if len(df) >= 4 else 0
#     gW3_xG = df.iloc[-3]['xG'] if len(df) >= 3 else 0
#     gW2_xG = df.iloc[-2]['xG'] if len(df) >= 2 else 0
#     gW1_xG = df.iloc[-1]['xG']
#     gW1_iCT = df.iloc[-1]['iCT']
#     CFi = ((gW4_xG * 0.7) + (gW3_xG * 0.8) + (gW2_xG * 0.9) + gW1_xG) + (gW1_iCT * .001)

#     return CFi







# def plot_histograms(df):
#     # list columns to plot
#     cols_to_plot = ['expected_goal_involvements', 'ict_index', 'goals_scored', 'assists']

#     for col in cols_to_plot:
#         sns.histplot(df[col], kde=True)
#         plt.title(f'Histogram of {col}')
#         plt.show()

# # run histogram on player data
# for player_id, player_df in player_data.items():
#     print(f"Player ID: {player_id}")
#     plot_histograms(player_df)

# concatenate all player dataframes into a single one
# all_players_df = pd.concat(player_data.values())

# # generate histograms
# for column in ['expected_goal_involvements', 'ict_index', 'goals_scored', 'assists']:
#     plt.figure(figsize=(10,8))
#     plt.hist(all_players_df[column], bins=80, edgecolor='black')
#     plt.title(f'Histogram of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')

#     # save plot as image file
#     plt.savefig(f'{column}_histogram.png')


# for player_id, player_df in player_data.items():
#     cfi = calculate_cfi(player_df)
#     print(f"CFi for player {player_id}: {cfi}")



# # inspect data of 1st player
# first_player_id = list(player_data.keys())[0]
# first_player_df = pd.DataFrame(player_data[first_player_id])

# # print datafram
# print(first_player_df)

# # look for missing values
# print("\nMissing values for each column:")
# print(first_player_df.isnull().sum())

#print out description for each player's dataframe
# for player_id, player_df in player_data.items():
#     print(f"\nPlayer ID: {player_id}")
#     print(player_df.describe())






























#player_url = 'https://fantasy.premierleague.com/api/element-summary/7/'


#player_data = get_data(player_url)


#print(player_data)
#print(player_data.keys())


#player_history_df = pd.DataFrame(player_data['history'])
#print(player_history_df.head())
#player_history_df.to_csv('/workspaces/43352219/final_project/test_player_df.csv', index=False)

#player_history_past_df = pd.DataFrame(player_data['history_past'])
#player_history_past_df.to_csv('/workspaces/43352219/final_project/test_player_history_past_df.csv', index=False)

#fixtures_past_df = pd.DataFrame(player_data['fixtures'])
#print(fixtures_past_df.head())
#fixtures_past_df.to_csv('/workspaces/43352219/final_project/fixtures_past_df_2.csv', index=False)





# get events json data & print to csv
# events_df = pd.DataFrame(data['events'])
# events_df.to_csv('events_data.csv', index=False)

# get phases json data & print to csv
# phases_df = pd.DataFrame(data['phases'])
# phases_df.to_csv('/workspaces/43352219/final_project/phases.csv', index=False)

# get teams json data & print to csv
# teams_df = pd.DataFrame(data['teams'])
# teams_df.to_csv('/workspaces/43352219/final_project/teams.csv', index=False)

# get total players & print to csv
# total_players_df = pd.DataFrame({'total_players': [data['total_players']]})
# total_players_df.to_csv('/workspaces/43352219/final_project/total_players.csv', index=False)

# get elements stats & print to csv
# element_stats_df = pd.DataFrame(data['element_stats'])
# element_stats_df.to_csv('/workspaces/43352219/final_project/element_stats.csv', index=False)

# get element types & print to csv
# element_types_df = pd.DataFrame(data['element_types'])
# element_types_df.to_csv('/workspaces/43352219/final_project/element_types.csv', index=False)










