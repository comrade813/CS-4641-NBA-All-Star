import time
from turtle import forward
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel, f_classif, mutual_info_classif, VarianceThreshold, RFE, SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

data_type = "advanced"
data = pd.read_csv(f"data/partially_processed/{data_type}_player_season_totals.csv")
data = data.drop(["name", "positions"], axis=1)

def features_only(data: pd.DataFrame):
    return data.drop("is_all_star", axis=1)

def get_selected_features(data: pd.DataFrame, support:list):
    features = data.columns[support].tolist()
    print(f"Features selected: {features}")
    return features

def get_correlation_heatmap(data: pd.DataFrame, threshold:float=0.5):
    plt.figure(figsize=(18,12))
    plt.subplots_adjust(bottom=0.3)   
    c = data.corr()
    _ = sb.heatmap(c, cmap="RdYlGn", annot=True)

    ranked = sorted([(i, j, c[i][j]**2) for i, j in itertools.combinations(c, 2) if not np.isnan(c[i][j])], key=lambda x: x[2], reverse=True)
    for i, j, val in ranked:
        if val < threshold:
            break
        print("{:<30}{:<30}{}".format(i, j, val))
    plt.show()

def get_mutual_information(x: pd.DataFrame, y:pd.Series):
    plt.figure(figsize=(10,5))
    plt.subplots_adjust(left=0.3)
    importances = mutual_info_classif(x, y)
    feat_importances = pd.Series(importances, x.columns)
    feat_importances.plot(kind="barh", color="teal")
    plt.show()

def get_variance(x: pd.DataFrame, threshold:float=0.1):
    tmp = x.drop(["games_played", "minutes_played"], axis=1)
    v_threshold = VarianceThreshold(threshold=threshold)
    v_threshold.fit(tmp)
    plt.figure(figsize=(8,5))
    plt.subplots_adjust(left=0.3)
    plt.barh(width=v_threshold.variances_, y=tmp.columns)
    plt.title("Variances of each feature")
    plt.show()
    get_selected_features(tmp, v_threshold.get_support())

def get_ridge(x: pd.DataFrame, y:pd.Series):
    ridge = RidgeClassifier().fit(x, y)
    importance = np.abs(ridge.coef_)[0]
    plt.figure(figsize=(8,5))
    plt.subplots_adjust(left=0.3)
    plt.barh(width=importance, y=x.columns)
    plt.title("Feature importances via coefficients")
    plt.show()

def get_estimator(x:pd.DataFrame, y:pd.Series, estimator:str, n:int):
    est = None
    if estimator == "ridge":
        est = RidgeClassifier().fit(x,y)
    elif estimator == "lsvc":
        est = LinearSVC(dual=False, max_iter=10000)
    elif estimator == "kneighbors":
        est = KNeighborsClassifier(n_neighbors=n)
    elif estimator == "kmeans":
        est = KMeans(n_clusters=n)
    return est

def forward_selection(x:pd.DataFrame, y:pd.Series, estimator:str="ridge", n:int=8):
    sfs = SequentialFeatureSelector(get_estimator(x, y, estimator,n), n_features_to_select=n, direction="forward")
    sfs.fit(x,y)
    return get_selected_features(x, sfs.get_support())

def backward_selection(x:pd.DataFrame, y:pd.Series, estimator:str="ridge", n:int=8):
    sfs = SequentialFeatureSelector(get_estimator(x, y, estimator,n), n_features_to_select=n, direction="backward")
    sfs.fit(x,y)
    return get_selected_features(x, sfs.get_support())

def forward_and_backward(x:pd.DataFrame, y:pd.Series):
    res = {}
    for key in x.columns:
        res[key] = 0

    for est in ["ridge", "lsvc", "kneighbors", "kmeans"]:
        selected = forward_selection(x, y, est)
        selected.extend(backward_selection(x, y, est))
        for key in selected:
            res[key] += 1
    
    keys = []
    count = []
    for key in res:
        keys.append(key)
        count.append(res[key])

    plt.figure(figsize=(8,5))
    plt.subplots_adjust(left=0.3)
    plt.barh(width=count, y=keys)
    plt.title("Feature importances via forward and backward selection")
    plt.show()

# only works with ridge and lsvc
def recursive_selection(x:pd.DataFrame, y:pd.Series, estimator:str="ridge"):
    print(x.shape, y.shape)
    rfe = RFE(estimator=get_estimator(x,y,estimator,8), n_features_to_select=1)
    rfe.fit(x,y)
    print(list(zip(rfe.ranking_, x.columns.values, rfe.get_support())))
    plt.figure(figsize=(8,5))
    plt.subplots_adjust(left=0.3)
    plt.barh(width=rfe.ranking_, y=x.columns)
    plt.title(f"Feature rankings via recursive selection with estimator {estimator}")
    plt.show()
    return get_selected_features(x, rfe.get_support())

def lasso_l1(x:pd.DataFrame, y:pd.Series):
    lscv = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=10000).fit(x,y)
    model = SelectFromModel(lscv, prefit=True)
    return get_selected_features(x, model.get_support())

def forest_feature_importance(x:pd.DataFrame, y:pd.Series):
    forest = RandomForestClassifier(random_state=0)
    forest.fit(x, y)
    importance = forest.feature_importances_
    plt.figure(figsize=(8,5))
    plt.subplots_adjust(left=0.3)
    plt.barh(width=importance, y=x.columns)
    plt.title("Feature importances via Random Forest Classifier")
    plt.show()

x, y = features_only(data), data["is_all_star"]
recursive_selection(x, y, "lsvc")

# STUFF TO DO
# ranking in forward selection