from re import X
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FastICA

features_to_drop = []
data_type = "advanced"

d_reduction_type = "PCA"

data = pd.read_csv(f"data/partially_processed/{data_type}_player_season_totals.csv")

x, y = data.drop(["name", "is_all_star"], axis=1), data["is_all_star"]
encoded = pd.get_dummies(x["positions"], prefix="POS_")
x = pd.merge(left=x.drop("positions", axis=1), right=encoded, left_index=True, right_index=True)

x = x.drop(features_to_drop, axis=1)

x, y = x.values, y.values

if d_reduction_type == "PCA":
    x = PCA(n_components=4).fit_transform(x)
elif d_reduction_type == "SVD":
    x = TruncatedSVD(n_components=4).fit_transform(x)
elif d_reduction_type == "ICA":
    x = FastICA(n_components=4)

data = np.hstack((x,y))

np.savetxt(f"data/fully_preprocessed/{data_type}.csv", data, delimiter=",")