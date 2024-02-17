# CFi Index (Catch Fire Index)
#### Video Demo: https://youtu.be/bvQTsiqKeb8
#### Description:

The CFi Index is a searchable player ranking system that applies a formula to player data in order to predict which players
have a higher likelihood of scoring in the following games in the Fantasy Premiere League, an online fantasy soccer league based on the English Premiere League. The FPL homepage can be found at: https://fantasy.premierleague.com/.

At the time of creating the program, the Premiere League was in the off season, and so the data used was a static request from last season's data.
The predictions and rankings were intended to rank the players at the end of last season.

As such, it began by making API request from the two FPL API's: https://fantasy.premierleague.com/api/element-summary/{player_id}/ and
https://fantasy.premierleague.com/api/bootstrap-static/. From there, the data was organized into a dictionary by player, and later into a pandas
dataframe. Afterwards, the data was used to calculate an initial 'Catch Fire Index' ranking, using

    (gW4_xG * 0.7) + (gW3_xG * 0.8) + (gW2_xG * 0.9) + gW1_xG + (iCT * 0.001)

as the intitial formula. gW4_xG represented the player's 'Expected Goals' or xG from three gameweeks ago, with gW3_xG being the same from two gameweeks ago, etc, and iCT being the 'creativity threat index' assigned to the player by the FPL. All of these values were pulled from the API requests.

After CFi was calculated, the data for goals, assists, iCT, CFi, xG, round (gameweek), team, position and player name was saved in the SQL database
fantasy_football.db for further testing. From here, the initial data was arranged into plots, histograms and heatmaps in order to look for correlations between the different variables.

After this, machine learning was applied using sklearn through linear regression and random forest testing, in order to further test the efficacy of the formula and train a predictive model. Linear regression proved ineffective at training the model, due to the data being based on an imbalanced class. The formula was eventually revised down to

    (gW3_xG * 0.8) + (gW2_xG * 0.9) + gW1_xG + (iCT * 0.001)

and the results of the random forest modeling were saved in a joblib file at fantasy_football_model.joblib. The revised data was then repopulated in the SQL database. The code to query the API, save into SQL and test using various methods can be found in the algo.py file.

After this, it was a matter of coding up a file to handle to retrieving the data from the SQL database, and another to display it on the front end. The app.py file handled the back end. The function:

```
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

```

covered the home page. The functions:

```

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

```

allowed for flexible querying of the SQL database depending on user selection. The function:

```
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
```

handled the operation of the filter to be selected by the user.

The index.html file handled the display of information on the front end. Bootstrap was used for the minimalist design implementation, linked to the styles.css file. The dropdown menus were displayed by:
```
<!-- Primary dropdown menu-->
                    <select class="custom-select mb-4" id="playerFilter" onchange="filterTypeChanged()">
                        <option selected>Choose...</option>
                        <option value="all">All Players</option>
                        <option value="position">Position</option>
                        <option value="team">Team</option>
                    </select>

                    <!--Secondary dropdown menus-->
                    <!--Team selection-->
                    <div id="teamDropdown" style="display: none;">
                        <select class="custom-select mb-4" id="teamFilter" onchange="filterPlayers()">
                            {% for key, team in team_mapping.items() %}
                            <option value="{{ key }}">{{ team }}</option>
                            {% endfor %}
                        </select>
                    </div>
                     <!--Position selection-->
                    <div id="positionDropdown" style="display: none;">
                        <select class="custom-select mb-4" id="positionFilter" onchange="filterPlayers()">
                            {% for key, position in position_mapping.items() %}
                            <option value="{{ key }}">{{ position }}</option>
                            {% endfor %}
                        </select>
                    </div>
```

The code to display the table containing the user-requested data can be found here:
```
<tbody id="dataTable">
                            {% for player in players %}
                            <tr>
                                <td>{{ player.player_name }}</td>
                                <td>{{ player.team }}</td>
                                <td>{{ player.position }}</td>
                                <td>{{ player.CFi }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                <div class="col">
                    <!--images go here-->
                </div>
            </div>
        </div>

```

Lastly, javascript functions were used to get the data from the back end and display the data on the front end interactively:
```
 function filterTypeChanged() {
                var selection = document.getElementById("playerFilter").value;

                // Hide both secondary dropdowns beforehand
                document.getElementById("teamDropdown").style.display = "none";
                document.getElementById("positionDropdown").style.display = "none";

                // Display dropdown based on selection
                if (selection == "team") {
                    document.getElementById("teamDropdown").style.display = "block";
                } else if (selection == "position") {
                    document.getElementById("positionDropdown").style.display = "block";
                } else if (selection == "all") {
                    filterPlayers();
                }
            }
            function filterPlayers() {
                var filterType = document.getElementById("playerFilter").value;

                var teamValue, positionValue;
                if (filterType === "team") {
                    teamValue = document.getElementById("teamFilter").value; //get selected team name
                } else if (filterType === "position") {
                    positionValue = document.getElementById("positionFilter").value; //get selected position name
                }



                $.ajax({
                    url: '/filter_players', //Flask route to handle filtering
                    type: 'POST',
                    data: {
                        'filter': filterType,
                        'team': teamValue,
                        'position': positionValue
                    },
                    dataType: 'json',
                    //update table in front end with new data
                    success: function(players_data) {
                        // Clear the existing table rows
                        var tbody = document.querySelector("table tbody");
                        tbody.innerHTML = '';

                        // Populate the table with new data
                        players_data.forEach(function(player) {
                            var tr = document.createElement('tr');

                            var tdName = document.createElement('td');
                            tdName.textContent = player.player_name;

                            var tdTeam = document.createElement('td');
                            tdTeam.textContent = player.team;

                            var tdPosition = document.createElement('td');
                            tdPosition.textContent = player.position;

                            var tdCFi = document.createElement('td');
                            tdCFi.textContent = player.CFi;

                            tr.appendChild(tdName);
                            tr.appendChild(tdTeam);
                            tr.appendChild(tdPosition);
                            tr.appendChild(tdCFi);

                            tbody.appendChild(tr);
                        });
                    }
                });
            }
```

