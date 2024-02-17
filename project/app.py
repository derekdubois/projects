from flask import Flask, render_template, request, jsonify
import pandas as pd
from sqlalchemy import create_engine

# create dictionary to map team numbers from API to team names in front end
team_mapping = {
    1: "Arsenal",
    2: "Aston Villa",
    3: "Bournemouth",
    4: "Brentford",
    5: "Brighton",
    6: "Chelsea",
    7: "Crystal Palace",
    8: "Everton",
    9: "Fulham",
    10: "Leicester City",
    11: "Leeds United",
    12: "Liverpool",
    13: "Manchester City",
    14: "Manchester United",
    15: "Newcastle United",
    16: "Nottingham Forest",
    17: "Southhampton",
    18: "Tottenham",
    19: "West Ham",
    20: "Wolves"
}

# create dictionary to maps position numbers from API to position names in front end
position_mapping = {
    2: "Defender",
    3: "Midfielder",
    4: "Forward",
}

# create reversed dictionaries for key:value pairs for teams and positions
team_mapping_reverse = {v:k for k,v in team_mapping.items()}
position_mapping_reverse = {v: k for k, v in position_mapping.items()}

# initialize Flask app
app = Flask(__name__)

# create a connection engine to SQLite database in current directory
engine = create_engine('sqlite:///fantasy_football.db')

# create function to query SQL database for latest gameweek info,
# convert to pandas dataframe, then return the first row & first column of the dataframe
def get_latest_gameweek(engine):
    MAX_GAMEWEEK_QUERY = "SELECT MAX(round) FROM gameweeks"
    return pd.read_sql(MAX_GAMEWEEK_QUERY, engine).iloc[0, 0]

# function for flexible SQL query based on position or team
def get_base_query_for_gameweek(gameweek, extra_condition=None):
    query = f"SELECT * FROM gameweeks WHERE round={gameweek}"
    # add extra condition if provided
    if extra_condition:
        query += f" AND {extra_condition}"

    # add ordering and limit clauses
    query += " ORDER BY CFi DESC LIMIT 25"

    return query


@app.route('/')
def home():
    latest_gameweek = get_latest_gameweek(engine)
    # load data from the SQLite database
    data = pd.read_sql(get_base_query_for_gameweek(latest_gameweek), engine)

    # convert dataframe to list of dictionaries for easier rendering in Flask
    data_records = data.to_dict(orient='records')

    # convert team and position numbers to corresponding names
    for record in data_records:
        record['team'] = team_mapping.get(record['team'], record['team'])
        record['position'] = position_mapping.get(record['position'], record['position'])

    return render_template('index.html', players=data_records, team_mapping=team_mapping, position_mapping=position_mapping)

# function to return all players from SQL
def get_all_players():

    latest_gameweek = get_latest_gameweek(engine)
    data = pd.read_sql(get_base_query_for_gameweek(latest_gameweek), engine)
    # convert pandas df to list of dictionaries
    data_records = data.to_dict(orient='records')

    for record in data_records: # try to map keys, if they don't exist, default to original value
        record['team'] = team_mapping.get(record['team'], record['team'])
        record['position'] = position_mapping.get(record['position'], record['position'])
    return data_records

# function to get players by position using flexible SQL query
def get_players_by_position(position_id):
    latest_gameweek = get_latest_gameweek(engine)
    query = get_base_query_for_gameweek(latest_gameweek, extra_condition=f"position={position_id}")

    data = pd.read_sql(query, engine)
    data_records = data.to_dict(orient='records')

    for record in data_records:
        record['team'] = team_mapping.get(record['team'], record['team'])
        record['position'] = position_mapping.get(record['position'], record['position'])
    return data_records


def get_players_by_team(team_id):
    latest_gameweek = get_latest_gameweek(engine)
    query = get_base_query_for_gameweek(latest_gameweek, extra_condition=f"team={team_id}")

    data = pd.read_sql(query, engine)
    data_records = data.to_dict(orient='records')

    for record in data_records:
        record['team'] = team_mapping.get(record['team'], record['team'])
        record['position'] = position_mapping.get(record['position'], record['position'])
    return data_records

# take java request & query database to return data to front end
@app.route('/filter_players', methods=['POST'])
def filter_players():
    filter_type = request.form.get('filter')
    team = int(request.form.get('team')) if request.form.get('team') else None
    position = int(request.form.get('position')) if request.form.get('position') else None

    #query database by filter
    if filter_type == 'all':
        players_data = get_all_players()
    elif filter_type == 'position' and position:
        players_data = get_players_by_position(position)
    elif filter_type == 'team' and team:
        players_data = get_players_by_team(team)
    else:
        players_data = []

    return jsonify(players_data)

if __name__ == '__main__':
    app.run(debug=True)