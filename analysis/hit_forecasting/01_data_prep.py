import sys
import os

import seaborn as sns
import pandas as pd
import numpy as np
import datetime as dt
from utilities import methods as utils

from sklearn.base import clone
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from utilities.mappings import maps as colmaps
from lifelines import KaplanMeierFitter, CoxPHFitter
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 200)

#
# Config
CONFIG = {
    'inpath': "/Volumes/Transcend/gameday/",
    'windowStartDate': dt.datetime(year=2015, month=4, day=1),
    'windowEndDate': dt.datetime(year=2015, month=11, day=1),
    'local': False,
    'lookback_games': 40,
    'lookforward_games': 10,
    'statistic': 'hits'
}

#
# Methods


if __name__ == "__main__":

    # Get subdirectories of gamedays
    subdirs = utils.get_gameday_subdirs(path=CONFIG.get('inpath'),
                                        window_start=CONFIG.get(
                                            'windowStartDate'),
                                        window_end=CONFIG.get(
                                            'windowEndDate'),
                                        left_incl=True, right_incl=True)

    # Append Files
    df_innings = pd.concat(
        objs=[pd.read_parquet(
            CONFIG.get('inpath') + subdir + '/innings.parquet')
              for subdir in subdirs if
              "innings.parquet" in os.listdir(
                  CONFIG.get('inpath') + subdir)],
        axis=0
    )

    # Get atbat level subset
    atbat_vars = [x for x in df_innings if x[:6] == 'atbat_']
    atbat_vars.append('game_id')
    df_atbats = df_innings.loc[:, atbat_vars]
    df_atbats.drop_duplicates(inplace=True)

    # Sort
    df_atbats.sort_values(
        by=['game_id', 'atbat_batter', 'atbat_num'],
        ascending=True,
        inplace=True
    )

    df_atbats = df_atbats.loc[:, ['game_id', 'atbat_batter', 'atbat_num'] + [
        x for x in df_atbats.columns if x not in [
            'game_id', 'atbat_batter', 'atbat_num'
        ]
    ]]

    # Get batters with greather than lookback+1 number of games
    batters_elig = df_atbats['atbat_batter'].value_counts().reset_index()
    batters_elig.columns = ['Batter', 'Appearances']
    elig_msk = (batters_elig['Appearances'] > CONFIG.get('lookback_games') + 1)
    batters_elig = list(pd.Series.unique(
        batters_elig.loc[elig_msk, :]['Batter']
    ))
    df_atbats = df_atbats.loc[df_atbats['atbat_batter'].isin(batters_elig), :]
    print("Remaining: {} unique batters over {} appearances".format(
        str(pd.Series.nunique(df_atbats['atbat_batter'])),
        str(df_atbats.shape[0])
    ))

    df_atbats.sort_values(by=['atbat_batter', 'atbat_num'],
                          ascending=True,
                          inplace=True)
    df_atbats.drop(labels=['atbat_num'], axis=1, inplace=True)

    # Add date to game
    df_atbats['game_date'] = df_atbats['atbat_start_tfs_zulu'].str[:10]
    df_atbats['game_date'] = pd.to_datetime(df_atbats['game_date'])

    df_atbats.to_pickle("/Users/peteraltamura/Desktop/atbats_prepped.pickle")


