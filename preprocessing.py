import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FastICA

# keep 10 features
features_to_keep = {"value_over_replacement_player", "true_shooting_percentage",
    "win_shares_per_48_minutes", "player_efficiency_rating", "usage_percentage", "offensive_box_plus_minus",
    "three_point_attempt_rate", "free_throw_attempt_rate", "steal_percentage"}
data_type = "advanced"

drop_features = "full"
normalized = "None"
dimensionality_reduction = "None"

data = pd.read_csv(f"data/partially_processed/{data_type}_player_season_totals.csv")

encoded = pd.get_dummies(data["positions"], prefix="POS")
data = pd.merge(left=data.drop("positions", axis=1), right=encoded, left_index=True, right_index=True)

X = data.drop("name", axis=1)
positions = ['POS_CENTER', 'POS_POINT GUARD', 'POS_POWER FORWARD', 'POS_SHOOTING GUARD', 'POS_SMALL FORWARD']
years = X['season'].unique()


if normalized == "normalized":
    working_X = []
    for pos in positions:
        for year in years:
            #get players for a particular year and find means and variances for every feature
            #current = X.where(X[pos] == 1 & X['season'] == year)

            current = X[(X[pos] == 1) & (X['season'] == year)]
            current = current.drop(columns=['season'])

            all_star_col = current['is_all_star']
            current = current.drop(columns=['is_all_star'])        

            position_cols = current[positions]
            current = current.drop(columns=positions)

            current = (current - current.mean())/current.std()
            current[positions] = position_cols
            current['is_all_star'] = all_star_col
            working_X.append(current)
    X = pd.concat(working_X)

if drop_features == "reduced":
    features_to_drop = list(set(X.keys()).difference(features_to_keep))
    x, y, names = X.drop(features_to_drop, axis=1), data["is_all_star"], data["name"]
elif normalized != "normalized":
    x, y, names = X.drop(["season", "is_all_star"]+positions, axis=1), data["is_all_star"], data["name"]
else:
    x, y, names = X.drop(["is_all_star"], axis=1), data["is_all_star"], data["name"]

n = 8
if dimensionality_reduction == "PCA":
    x = pd.DataFrame(PCA(n_components=n).fit_transform(x), columns=[f"pca_{i}" for i in range(0,n)])
elif dimensionality_reduction == "SVD":
    x = TruncatedSVD(n_components=n).fit_transform(x)
elif dimensionality_reduction == "ICA":
    x = FastICA(n_components=n)

data = pd.concat([names, x, y], axis=1)

data.to_csv(f"data/fully_processed/{drop_features}_{normalized}_{dimensionality_reduction}.csv", sep=",", index=False)