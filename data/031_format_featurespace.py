import os
import gc
import sys
import numpy as np
import pandas as pd
import datetime as dt
import utilities as util

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
CONFIG = util.load_config()


def read_in_and_filter(paths, idx_cols, target):
    """
    """

    # List to concatenate later
    df_all_year = []
    last_col_set = None

    # Begin Iteration
    for path_ in paths:

        # Read in 
        df = pd.read_parquet(path_)

        # Determine metric cols
        metric_cols = [
            x for x in df.columns if
            any(p in x for p in pitcher_metrics) or
            any(b in x for b in batter_metrics)
        ]

        # Determine metric cols meeting fill thresh
        metric_cols = [
            m for m in metric_cols if sum(df[m].notnull()) > min_fill_thresh
        ]

        # Subset to just those metrics plus target and index
        df = df.loc[:,
            idx_cols + metric_cols + [target]
        ]

        # Fill with median
        for col in metric_cols:
            df[col].fillna(np.nanmedian(df[col]), inplace=True)

        # Stack
        df_all_year.append(df)

        if last_col_set == None:
            last_col_set = sorted(list(df.columns))
        else:
            if set(last_col_set) != set(df.columns):
                print(path_)
                print(sorted(df.columns))
                print(len([x for x in last_col_set if x not in df.columns]))
                print(len([x for x in df.columns if x not in last_col_set]))
                raise

    # Concatenate
    df_all_year = pd.concat(
        objs=df_all_year,
        axis=0
    )
    return df_all_year
        


def match_games(data):
    """
    """

    # Get list of gameIds with 2 observations
    gameIdVC = data['gameId'].value_counts()
    gameIdVC = pd.DataFrame(gameIdVC)
    gameIdVC.reset_index(inplace=True)
    gameIdVC.columns = ['gameId', 'freq']
    gameIdVC = gameIdVC.loc[gameIdVC['freq'] == 2, :]

    matched = []
    # Iterate
    for gid in list(set(gameIdVC['gameId'])):
        home = data.loc[(
            (data['gameId'] == gid)
            &
            (data['batterTeamFlag'] == 'home')
            &
            (data['pitcherTeamFlag'] == 'home')
        ), :]
        away = data.loc[(
            (data['gameId'] == gid)
            &
            (data['batterTeamFlag'] == 'away')
            &
            (data['pitcherTeamFlag'] == 'home')
        ), :]
        df = pd.merge(
            home,
            away,
            how='left',
            on=['gameId'],
            suffixes=['_home', '_away'],
            validate='1:1'
        )
        matched.append(df)
    matched = pd.concat(
        objs=matched,
        axis=0
    )
    return matched


def rename_drop_dimensions(date):
    """
    """

    data.drop(
        labels=['home_team_win_pct_at_home_home',
                'away_team_win_pct_at_away_home',
                'home_team_winner_home'],
        axis=1,
        inplace=True
    )
    data.rename(
        columns={'home_team_win_pct_at_home_away': 'home_team_win_pct_at_home',
                 'away_team_win_pct_at_away_away': 'away_team_win_pct_at_away',
                 'home_team_winner_away': 'home_team_winner'},
        inplace=True
    )
    return df_matchups


def scale_features(train_games, test_games):
    """
    """
    
    #Instantiate
    scaler = StandardScaler()

    # Fit on training set only.
    scaler.fit(train_games)

    # Apply transform to both the training set and the test set.
    train_games = scaler.transform(train_games)
    test_games = scaler.transform(test_games)
    return train_games, test_games


def apply_pca(train_games, test_games):
    """
    """

    # Instantiate pca object
    pca = PCA(.95)

    # Fit on train
    pca.fit(train_img)

    # Transform train and test
    train_games = pca.transform(train_games)
    test_games = pca.transform(test_games)

    return train_games, test_games


if __name__ == "__main__":

    # Full dataset path
    years = ['2017']
    min_fill_thresh = 0.96
    test_size = 0.3
    
    top_pitcher_count = 15
    pitcher_metrics = ['BF', 'ER', 'ERA', 'HitsAllowed', 'Holds',
                       'SeasonLosses', 'SeasonWins', 'numberPitches',
                       'Outs', 'RunsAllowed', 'Strikes', 'SO']
    top_batter_count = 15
    batter_metrics = ['Assists', 'AB', 'BB', 'FO', 'Avg', 'H',
                      'HBP', 'HR', 'Doubles' 'GroundOuts', 'batterLob',
                      'OBP', 'OPS', 'R', 'RBI', 'SluggingPct',
                      'StrikeOuts', 'Triples']    

    # Index and Metrics
    idx_cols = [
        'home_team_win_pct_at_home', 'away_team_win_pct_at_away',
        'home_team_winner'
    ]
    target = 'home_team_winner'
    
    # File list
    filelist_year = [
        CONFIG.get('paths').get('initial_featurespaces')+fname
        for fname in os.listdir(
            CONFIG.get('paths').get('initial_featurespaces')
        ) if any(y in fname for y in years)
    ]

    # Get all year data table together
    df_all_year = read_in_and_filter(filelist_year,
                                     idx_cols=idx_cols,
                                     target=target)

    # Match Games
    df_matchups = match_games(df_all_year)
    
    # Filter Columns
    df_matchups = rename_drop_dimensions(df_matchups)

    # Train Test Split
    train_games, test_games, train_result, test_result = \
        train_test_split(df_matchups, df_targets, test_size=test_size, random_state=0)

    # Scale before PCA
    train_games, test_games = scale_features(train_games, test_games)

    # Run PCA
    train_games, test_games = apply_pca(train_games, test_games)
    
    
