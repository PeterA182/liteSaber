import sys
import os

import seaborn as sns
import pandas as pd
import numpy as np
import datetime as dt
#from utilities import methods as utils

from sklearn.base import clone
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from a_data_prep import CONFIG
#from utilities.mappings import maps as colmaps
from lifelines import KaplanMeierFitter, CoxPHFitter
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 200)

#
# ---- ---- ----
def add_metrics(data):
    """
    Adds metrics from given lookback_data and pairs with lookforward
    statistic of measure

    PARAMETERS
    ----------
    lookback_data: DataFrame
        contains atbat observations from lookback window defined in CONFIG
    lookforward_data: DataFram
        contains atbat observations to calculate statistic of measure from
        and pair to lookback observations
    """

    # Add Hits metrics
    hit_events = [
        'Single', 'Triple', 'Double', 'Home Run'
    ]
    hit_msk = (data['atbat_event'].isin(hit_events))
    data['hit_flag'] = hit_msk.astype(int)

    # Add On Base metrics
    on_base_events_plus_hr = [
        'Single', 'Walk', 'Triple', 'Home Run', 'Hit By Pitch',
        'Intent Walk'
    ]
    on_base_events_plus_hr_msk = (
        data['atbat_event'].isin(on_base_events_plus_hr)
    )
    data['on_base_plus_hr_flag'] = \
        on_base_events_plus_hr_msk.astype(int)

    # Produced out
    produced_out = [
        'Strikeout', 'Flyout', 'Groundout', 'Pop Out', 'Grounded Into DP',
        'Fielders Choice Out', 'Forceout', 'Lineout', 'Double Play',
        'Bunt Pop Out', 'Bunt Groundout', 'Strikeout - DP', 'Bunt Lineout',
        'Triple Play'
    ]
    produced_out_msk = (data['atbat_event'].isin(produced_out))
    data['produced_out_flag'] = produced_out_msk.astype(int)

    return data


if __name__ == "__main__":

    # Read in prepped atbats
    df_atbats = pd.read_pickle(path="/Users/peteraltamura/Desktop/atbats_prepped.pickle")

    # TODO - this should be made much more robust for missed games, etc
    # Iterate over each slice of x back and y forward games in remaining data
    lb_sequences = []
    lf_sequences = []
    for start_date in list(pd.Series.unique(df_atbats['game_date'])):

        # Create current sequence number to tie lookback and lookforward later
        seq_number = "a_{}".format(str(start_date))

        # All games after current date
        atbat_slice = df_atbats.loc[df_atbats['game_date'] >= start_date, :]

        # Filter to players with (lb + lf) periods left
        batter_vc = atbat_slice['atbat_batter'].value_counts().reset_index()
        batter_vc.columns = ['Batter', 'Appearances']
        elig_batter_list = list(pd.unique(batter_vc.loc[
                                          batter_vc['Appearances'] >= (
                                                  CONFIG.get('lookback_games') +
                                                  CONFIG.get(
                                                      'lookforward_games')
                                          ), :]['Batter']))
        atbat_slice = atbat_slice.loc[
                      atbat_slice['atbat_batter'].isin(elig_batter_list), :]
        if atbat_slice.shape[0] == 0:
            break

        # Assign appearance numbers
        atbat_slice['appearance_num'] = atbat_slice.groupby(
            by=['atbat_batter'],
            as_index=False
        )['game_date'].rank(ascending=True, method='first')
        atbat_slice.loc[:, 'appearance_num'] = \
            atbat_slice['appearance_num'].astype(int)
        atbat_slice['sequence_number'] = seq_number

        # Split
        lb_sequences.append(
            atbat_slice.loc[
            atbat_slice['appearance_num'] < CONFIG.get('lookback_games'), :]
        )
        lf_sequences.append(
            atbat_slice.loc[
            atbat_slice['appearance_num'] > CONFIG.get('lookforward_games'), :]
        )
    print("Lookback Sequences")
    print(len(lb_sequences))
    print("Lookforward Sequences")
    print(len(lf_sequences))

    lb_sequences = pd.concat(
        objs=lb_sequences,
        axis=0
    )
    lf_sequences = pd.concat(
        objs=lf_sequences,
        axis=0
    )

    # Add Metrics
    lb_sequences_metrics = add_metrics(data=lb_sequences)
    lf_sequences_metrics = add_metrics(data=lf_sequences)

    # Send each to pickle

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # Pair lookforward hits with lookback metrics

    # Aggregate Lookback Games and Lookforward Games
    # LB
    lb_sequences_metrics = lb_sequences_metrics.groupby(
        by=['sequence_number', 'atbat_batter'],
        as_index=False
    ).agg({'atbat_away_team_runs': np.mean,
           'atbat_b': np.mean,
           'atbat_home_team_runs': np.mean,
           'atbat_o': np.mean,
           'atbat_s': np.mean,
           'hit_flag': np.sum})
    lb_sequences_metrics.rename(
        columns={x: x + "_mean" for x in list(lb_sequences_metrics.columns)
                 if x not in ['sequence_number', 'atbat_batter']},
        inplace=True
    )

    # LF
    lf_sequences_metrics = lf_sequences_metrics.groupby(
        by=['sequence_number', 'atbat_batter'],
        as_index=False
    ).agg({'hit_flag': np.sum})

    # Merge by atbat_batter and sequence_id
    seq_pairings = pd.merge(
        lb_sequences_metrics,
        lf_sequences_metrics,
        how='left',
        on=['sequence_number', 'atbat_batter']
    )

    # plot the heatmap
    corr = seq_pairings.corr()
    hm = sns.heatmap(corr,
                     xticklabels=corr.columns,
                     yticklabels=corr.columns)
    fig = hm.get_figure()
    fig.savefig("/Users/peteraltamura/Desktop/hit_flag_heatmap.png")

    # Split Exogenous and Endogenous
    X = seq_pairings.loc[:, [
                                'atbat_away_team_runs_mean', 'atbat_b_mean',
                                'atbat_home_team_runs_mean', 'atbat_o_mean',
                                'atbat_s_mean', 'hit_flag_mean'
                            ]]
    y = seq_pairings['hit_flag']

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    fold_iter = 0
    RANDOM_SEED = 1
    N_FOLDS = 10
    cv_results = {}
    skfolds = KFold(n_splits=N_FOLDS, shuffle=False, random_state=RANDOM_SEED)
    rfc = RandomForestRegressor(max_features='sqrt',
                                bootstrap=True,
                                n_estimators=10)

    for train_index, test_index in skfolds.split(X):
        # Prep exogenous
        x_train_fold = X.iloc[train_index, :]
        x_test_fold = X.iloc[test_index, :]

        # Prep endogenous
        y_train_fold = y.iloc[train_index]
        y_test_fold = y.iloc[test_index]

        # RF Model
        rf_curr_model = clone(rfc)
        rf_curr_model.fit(x_train_fold, y_train_fold)
        yhat = rf_curr_model.predict(x_test_fold)
        cod = r2_score(y_test_fold, yhat)
        cv_results[("Full", fold_iter)] = cod
        fold_iter += 1

    # Average COD
    cod_mean = np.mean(list(cv_results.values()))
    print(cod_mean)