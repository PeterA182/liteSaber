import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold, StratifiedKFold

# Read in
path = '/Volumes/Samsung_T5/mlb/gdApi/99_hr_prop_ft/featurespace_all.parquet'
df_ft = pd.read_parquet(path)

# Temp - replace BABIP trail
df_ft.loc[df_ft['batterBABIP_trail3'] == np.inf, 'batterBABIP_trail3'] = 0

X_idx = df_ft.loc[:, ['gameId', 'gameDate', 'atbat_batter', 'atbat_opp_starter', 'team']]
X = df_ft.loc[:, [
    x for x in df_ft.columns if x not in X_idx.columns and x != 'hr_flag'
]]
y = df_ft['hr_flag']

# Model
print(X_idx.shape)
print(X.shape)
print(y.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

print("--------")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

rf_clf = RandomForestClassifier(criterion='gini',
                                n_estimators=45,
                                min_samples_split=4,
                                min_samples_leaf=2,
                                max_features=0.33,
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)
rf_clf.fit(X_train, y_train)
yhat = rf_clf.predict(X_test)
pp = rf_clf.predict_proba(X_test)[:, 1]
f1_score_rf = f1_score(y_test, yhat)
acc_rf = accuracy_score(y_test, yhat)

print("F1 Score")
print(f1_score_rf)
print("Acc Score")
print(acc_rf)

df_comp = pd.DataFrame({
    'yhat': list(yhat),
    'y_test': list(y_test),
    'prob': list(pp)
})
df_comp.to_csv('/Users/peteraltamura/Desktop/rf_clf_hit_prop.csv', index=False)
df_comp['0_pred_correct'] = (
    (df_comp['yhat'] == 0)
    &
    (df_comp['y_test'] == 0)
)
df_comp['1_pred_correct'] = (
    (df_comp['yhat'] == 1)
    &
    (df_comp['y_test'] == 1)
)
df_comp['0_pred_incorrect'] = (
    (df_comp['yhat'] == 0)
    &
    (df_comp['y_test'] == 1)
)
df_comp['1_pred_incorrect'] = (
    (df_comp['yhat'] == 1)
    &
    (df_comp['y_test'] == 0)
)
print("Avg 0_pred_correct")
print(np.mean(df_comp['0_pred_correct']))
print("Avg 1_pred_correct")
print(np.mean(df_comp['1_pred_correct']))
print("Avg 0_pred_incorrect")
print(np.mean(df_comp['0_pred_incorrect']))
print("Avg 1_pred_incorrect")
print(np.mean(df_comp['1_pred_incorrect']))
print("---- ---- ---- ----\n"*4)

# GridSearch
param_grid = [
        {'n_estimators': [7, 8, 10], 
         'max_features': [0.1, 0.3, 0.4], 
         'min_samples_split': [10], 
         'min_samples_leaf': [10, 12, 15],
         'n_jobs': [-1]}]
rf_reg = RandomForestClassifier()
grid_search = GridSearchCV(rf_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(X_train, y_train)
print(param_grid)
print(grid_search.best_params_)

rf_clf_optimal = RandomForestClassifier(
    n_estimators=grid_search.best_params_.get('n_estimators'), 
    criterion='gini',
    max_depth=None,
    min_samples_split=grid_search.best_params_.get('min_samples_split'),
    min_samples_leaf=grid_search.best_params_.get('min_samples_leaf'),
    max_features=grid_search.best_params_.get('max_features')
)
rf_clf_optimal.fit(X_train, y_train)
yhat = rf_clf_optimal.predict(X_test)
pp = rf_clf_optimal.predict_proba(X_test)[:, 1]
f1_score_rf = f1_score(y_test, yhat)
acc_rf = accuracy_score(y_test, yhat)

print("F1 Score")
print(f1_score_rf)
print("Acc Score")
print(acc_rf)

df_comp = pd.DataFrame({
    'yhat': list(yhat),
    'y_test': list(y_test),
    'prob': list(pp)
})
df_comp.to_csv('/Users/peteraltamura/Desktop/rf_clf_hit_prop_optimal.csv', index=False)
df_comp['0_pred_correct'] = (
    (df_comp['yhat'] == 0)
    &
    (df_comp['y_test'] == 0)
)
df_comp['1_pred_correct'] = (
    (df_comp['yhat'] == 1)
    &
    (df_comp['y_test'] == 1)
)
df_comp['0_pred_incorrect'] = (
    (df_comp['yhat'] == 0)
    &
    (df_comp['y_test'] == 1)
)
df_comp['1_pred_incorrect'] = (
    (df_comp['yhat'] == 1)
    &
    (df_comp['y_test'] == 0)
)

