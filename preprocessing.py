from re import X
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FastICA

# keep 10 features
features_to_keep = {"value_over_replacement_player", "true_shooting_percentage", "positions",
    "win_shares_per_48_minutes", "player_efficiency_rating", "usage_percentage", "offensive_box_plus_minus"}
data_type = "advanced"
output_data_type = "PCA"
drop_features = "full"

data = pd.read_csv(f"data/partially_processed/{data_type}_player_season_totals.csv")
if drop_features == "reduced":
    features_to_drop = list(set(data.keys()).difference(features_to_keep))
    x, y, names = data.drop(features_to_drop, axis=1), data["is_all_star"], data["name"]
else:
    x, y, names = data.drop(["name", "is_all_star"], axis=1), data["is_all_star"], data["name"]
encoded = pd.get_dummies(x["positions"], prefix="POS_")
x = pd.merge(left=x.drop("positions", axis=1), right=encoded, left_index=True, right_index=True)

if output_data_type == "PCA":
    x = pd.DataFrame(PCA(n_components=4).fit_transform(x), columns=[f"pca_{i}" for i in range(0,4)])
elif output_data_type == "SVD":
    x = TruncatedSVD(n_components=4).fit_transform(x)
elif output_data_type == "ICA":
    x = FastICA(n_components=4)

print(names.shape, x.shape, y.shape)
data = pd.concat([names, x, y], axis=1)
print(data.shape)

data.to_csv(f"data/fully_processed/{drop_features}_{output_data_type}.csv", sep=",", index=False)