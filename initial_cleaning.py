from numpy import delete
import pandas as pd

all_stars = pd.read_csv("data/all_star_rosters.csv")

data_type = "advanced"
file_prefix = f"raw_{data_type}/"
file_suffix = f"_{data_type}_player_season_totals.csv"

all_data = pd.DataFrame()

for year in all_stars.keys():
    data = pd.read_csv("data/" + file_prefix + year + file_suffix)
    data = data.drop(["slug", "team", "is_combined_totals"], 1)

    players = dict()
    delete_rows = []

    sum_col = ['games_played', 'minutes_played']
    avg_col = ['player_efficiency_rating', 'true_shooting_percentage',
       'three_point_attempt_rate', 'free_throw_attempt_rate',
       'offensive_rebound_percentage', 'defensive_rebound_percentage',
       'total_rebound_percentage', 'assist_percentage', 'steal_percentage',
       'block_percentage', 'turnover_percentage', 'usage_percentage',
       'offensive_win_shares', 'defensive_win_shares', 'win_shares',
       'win_shares_per_48_minutes', 'offensive_box_plus_minus',
       'defensive_box_plus_minus', 'box_plus_minus',
       'value_over_replacement_player']

    expected = []
    all_star_team = set(all_stars[year].values.flatten())

    for i in range(0, len(data)):
        name = data["name"][i]
        if name in players:
            row, count = players[name]
            for key in sum_col:
                data[key][row] += data[key][i]
            for key in avg_col:
                data[key][row] = round((data[key][row]*count + data[key][i])/(count+1), 2)
            players[name] = (row, count+1)
            delete_rows.append(i)
        else:
            players[name] = (i, 1)
            if name in all_star_team:
                expected.append(1)
            else:
                expected.append(0)
    data = data.drop(delete_rows)

    #encode season that player corresponds to
    data['season'] = [year] * len(data)

    data["is_all_star"] = expected
    
    all_data = all_data.append(data)

all_data.to_csv("data/partially_processed/" + file_suffix[1:], index=False)