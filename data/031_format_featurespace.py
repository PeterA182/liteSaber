import os
import gc
import sys
import numpy as np
import pandas as pd
import datetime as dt
import utilities as util
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
CONFIG = util.load_config()


def read_in_and_filter(paths, idx_cols, target):
    """
    """

    # List to concatenate later
    df_all_year = []
    metrics_final = []

    # Begin Iteration
    for path_ in paths:
        print(path_)

        # Read in 
        df = pd.read_parquet(path_)
        pitch_fill = {col: np.mean(df[col].notnull()) for col in [x for x in df.columns if 'pitch' in x]}
        pitch_full = pd.DataFrame({'Col': list(pitch_fill.keys()), 'Fill': list(pitch_fill.values())})
        pitch_full.to_csv('/Users/peteraltamura/Desktop/pitch_fills.csv')
        
        # Determine metric cols
        metric_cols = [
            x for x in df.columns if
            any(p in x for p in pitcher_metrics) or
            any(b in x for b in batter_metrics)
        ]

        # Determine metric cols meeting fill thresh
        metric_bat_cols = [
            m for m in metric_cols if ('batter' in m) and (np.mean(df[m].notnull()) > min_fill_thresh_batting)
        ]
        metric_pitch_cols = [
            m for m in metric_cols if ('pitcher' in m) and (np.mean(df[m].notnull()) > min_fill_thresh_pitching)
        ]
        metric_cols = metric_bat_cols + metric_pitch_cols

        if len(metrics_final) == 0:
            metrics_final.extend(metric_cols)
        
        # Subset to just those metrics plus target and index
        df = df.loc[:,
            idx_cols + metrics_final
        ]

        # Fill with median
        for col in metrics_final:
            if np.mean(df[col].isnull()) == 1:
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna(np.nanmedian(df[col]), inplace=True)

        # Stack
        df_all_year.append(df)
        sort_cols = sorted(list(df.columns))
        assert 'home_team_winner' in df.columns

    df_all_year_final = []
    for df in df_all_year:
        df_all_year_final.append(df[idx_cols + metrics_final])

    # Concatenate
    df_all_year = pd.concat(
        objs=df_all_year_final,
        axis=0
    )
    return df_all_year
        


def match_games(data):
    """
    """
    print(data.shape)
    # Get list of gameIds with 2 observations
    ids = data['gameId'].value_counts()
    print(ids)
    ids = pd.DataFrame(ids).reset_index(inplace=False)
    ids.columns = ['gameId', 'freq']
    print(ids.shape)
    ids = ids.loc[ids['freq'] == 2, :]
    print(ids)
    matched = []
    # Iterate
    for gid in list(set(ids['gameId'])):
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
            (data['pitcherTeamFlag'] == 'away')
        ), :]
        df = pd.merge(
            home,
            away,
            how='inner',
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


def rename_drop_dimensions(data):
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
    return data


def scale_features(train_games, test_games):
    """
    """
    
    #Instantiate
    scaler = MinMaxScaler()

    # Fit on training set only.
    train_game_ids = train_games.loc[:, ['gameId',  'away_team_win_pct_at_away', 'home_team_win_pct_at_home']]
    test_game_ids = test_games.loc[:, ['gameId',  'away_team_win_pct_at_away', 'home_team_win_pct_at_home']]
    train_games = train_games.drop(labels=['gameId',  'away_team_win_pct_at_away', 'home_team_win_pct_at_home'], axis=1, inplace=False)
    test_games = test_games.drop(labels=['gameId',  'away_team_win_pct_at_away', 'home_team_win_pct_at_home'], axis=1, inplace=False)
    scaler.fit(train_games)
    train_games.to_csv('/Users/peteraltamura/Desktop/train_pre_ss.csv', index=False)
    
    # Apply transform to both the training set and the test set.
    train_games = pd.DataFrame(scaler.transform(train_games),
                               columns=train_games.columns)
    test_games = pd.DataFrame(scaler.transform(test_games),
                              columns=test_games.columns)
    train_games.to_csv('/Users/peteraltamura/Desktop/train_post_ss.csv', index=False)
    
#    train_games = scaler.transform(train_games)
#    test_games = scaler.transform(test_games)

    # Recombine
    train_games = train_game_ids.join(train_games)
    test_games = test_game_ids.join(test_games)

    return train_games, test_games


def apply_pca(train_games, test_games, pca_pct):
    """
    """

    # Split label gameIds
    train_game_ids = train_games.loc[:, ['gameId', 'away_team_win_pct_at_away', 'home_team_win_pct_at_home']]
    test_game_ids = test_games.loc[:, ['gameId', 'away_team_win_pct_at_away', 'home_team_win_pct_at_home']]
    train_games.drop(labels=['gameId', 'away_team_win_pct_at_away', 'home_team_win_pct_at_home'], axis=1, inplace=True)
    test_games.drop(labels=['gameId', 'away_team_win_pct_at_away', 'home_team_win_pct_at_home'], axis=1, inplace=True)
    train_games.to_csv('/Users/peteraltamura/Desktop/pre_pca_train.csv', index=False)
    
    # --------------------
    # Batting
    # Home
    pca = PCA()
    train_games_pca_home_bat = pca.fit_transform(train_games[[x for x in train_games.columns if ((x[:6] == 'batter') & (x[-5:] == '_home'))]])
    train_games_pca_home_bat = pd.DataFrame(train_games_pca_home_bat)
    train_games_pca_home_bat.columns = ['PCA_home_batting{}'.format(str(i)) for i in range(train_games_pca_home_bat.shape[1])]
    test_games_pca_home_bat = pca.transform(test_games[[x for x in test_games.columns if ((x[:6] == 'batter') & (x[-5:] == '_home'))]])
    test_games_pca_home_bat = pd.DataFrame(test_games_pca_home_bat)
    test_games_pca_home_bat.columns = ['PCA_home_batting{}'.format(str(i)) for i in range(test_games_pca_home_bat.shape[1])]
    train_games_pca_home_bat.to_csv('/Users/peteraltamura/Desktop/home_bat.csv', index=False)
    # Away
    pca = PCA()
    train_games_pca_away_bat = pca.fit_transform(train_games[[x for x in train_games.columns if ((x[:6] == 'batter') & (x[-5:] == '_away'))]])
    train_games_pca_away_bat = pd.DataFrame(train_games_pca_away_bat)
    train_games_pca_away_bat.columns = ['PCA_away_batting{}'.format(str(i)) for i in range(train_games_pca_away_bat.shape[1])]
    test_games_pca_away_bat = pca.transform(test_games[[x for x in test_games.columns if ((x[:6] == 'batter') & (x[-5:] == '_away'))]])
    test_games_pca_away_bat = pd.DataFrame(test_games_pca_away_bat)
    test_games_pca_away_bat.columns = ['PCA_away_batting{}'.format(str(i)) for i in range(test_games_pca_away_bat.shape[1])]

    # --------------------
    # Pitching
    # Home
    pca = PCA()
    train_games_pca_home_pitch = pca.fit_transform(train_games[[x for x in train_games.columns if ((x[:7] == 'pitcher') & (x[-5:] == '_home'))]])
    train_games_pca_home_pitch = pd.DataFrame(train_games_pca_home_pitch)
    train_games_pca_home_pitch.columns = ['PCA_home_pitching{}'.format(str(i)) for i in range(train_games_pca_home_pitch.shape[1])]
    test_games_pca_home_pitch = pca.transform(test_games[[x for x in test_games.columns if ((x[:7] == 'pitcher') & (x[-5:] == '_home'))]])
    test_games_pca_home_pitch = pd.DataFrame(test_games_pca_home_pitch)
    test_games_pca_home_pitch.columns = ['PCA_home_pitching{}'.format(str(i)) for i in range(test_games_pca_home_pitch.shape[1])]
        
    # Away
    pca = PCA()
    train_games_pca_away_pitch = pca.fit_transform(train_games[[x for x in train_games.columns if ((x[:7] == 'pitcher') & (x[-5:] == '_away'))]])
    train_games_pca_away_pitch = pd.DataFrame(train_games_pca_away_pitch)
    train_games_pca_away_pitch.columns = ['PCA_away_pitching{}'.format(str(i)) for i in range(train_games_pca_away_pitch.shape[1])]
    test_games_pca_away_pitch = pca.transform(test_games[[x for x in test_games.columns if ((x[:7] == 'pitcher') & (x[-5:] == '_away'))]])
    test_games_pca_away_pitch = pd.DataFrame(test_games_pca_away_pitch)
    test_games_pca_away_pitch.columns = ['PCA_away_pitching{}'.format(str(i)) for i in range(test_games_pca_away_pitch.shape[1])]
    
    # Recombine
    # Home
    train_games = train_game_ids.join(train_games_pca_home_bat)
    train_games = train_games.join(train_games_pca_away_bat)
    train_games = train_games.join(train_games_pca_home_pitch)
    train_games = train_games.join(train_games_pca_away_pitch)

    test_games = test_game_ids.join(test_games_pca_home_bat)
    test_games = test_games.join(test_games_pca_away_bat)
    test_games = test_games.join(test_games_pca_home_pitch)
    test_games = test_games.join(test_games_pca_away_pitch)

    return train_games, test_games


def select_features(train_games, test_games, train_result, test_result):
    """
    """

    # Split label gameIds
    train_game_ids = train_games.loc[:, ['gameId', 'away_team_win_pct_at_away', 'home_team_win_pct_at_home']]
    test_game_ids = test_games.loc[:, ['gameId', 'away_team_win_pct_at_away', 'home_team_win_pct_at_home']]
    train_games.drop(labels=['gameId', 'away_team_win_pct_at_away', 'home_team_win_pct_at_home'], axis=1, inplace=True)
    test_games.drop(labels=['gameId', 'away_team_win_pct_at_away', 'home_team_win_pct_at_home'], axis=1, inplace=True)
    train_games.to_csv('/Users/peteraltamura/Desktop/pre_pca_train.csv', index=False)
    
    # Set X and Y
    sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
    sel.fit(train_games, train_result)
    selected_feat = train_games.columns[(sel.get_support())]
    train_games = train_games.loc[:, selected_feat]

    # Filter train and test
    return list(selected_feat)


if __name__ == "__main__":

    # Full dataset path
    years = ['2017']
    min_fill_thresh_batting = 0.80
    min_fill_thresh_pitching = 0.35
    test_size = 1/3.0
    pca_pct = .80
    
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
        'gameId', 
        'home_team_win_pct_at_home', 'away_team_win_pct_at_away',
        'home_team_winner', 'batterTeamFlag', 'pitcherTeamFlag'
    ]
    target = 'home_team_winner'

    for yr in years:
        print("{} games now being prepared".format(yr))

        # File list
        filelist_year = [
            CONFIG.get('paths').get('initial_featurespaces')+fname
            for fname in os.listdir(
                CONFIG.get('paths').get('initial_featurespaces')
            ) if yr in fname
        ]

        # Get all year data table together
        df_all_year = read_in_and_filter(filelist_year,
                                         idx_cols=idx_cols,
                                         target=target)

        # Match Games
        df_matchups = match_games(df_all_year)

        # Filter Columns
        df_matchups = rename_drop_dimensions(df_matchups)

        # Set targets dataframe
        df_targets = df_matchups[['home_team_winner']]
        df_matchups.drop(labels=['home_team_winner'], axis=1, inplace=True)

        # Drop strings
        df_matchups.drop(
            labels=[x for x in df_matchups.columns if any(
                flag in x for flag in ['batterTeamFlag', 'pitcherTeamFlag']
            )],
            axis=1,
            inplace=True
        )                
        
        # Train Test Split
        train_games, test_games, train_result, test_result = \
            train_test_split(df_matchups, df_targets, test_size=test_size, random_state=0)
        
        # Select Features
        selected_feats = \
            select_features(train_games, test_games, train_result, test_result)
        
        # Save
        train_games = train_games.loc[:, selected_feats]
        test_games = test_games.loc[:, selected_feats]
        train_games.to_parquet(
            CONFIG.get('paths').get('full_featurespaces') + \
            '{}_train_full_featurespace.parquet'.format(str(yr))
        )
        test_games.to_parquet(
            CONFIG.get('paths').get('full_featurespaces') + \
            '{}_test_full_featurespace.parquet'.format(str(yr))
        )
