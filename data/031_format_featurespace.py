import os
import gc
import sys
import numpy as np
import pandas as pd
import datetime as dt
import utilities as util
CONFIG = util.load_config()


def read_in_and_filter(paths):
    """
    """

    # List to concatenate later
    df_all_year = []

    # Begin Iteration
    for path_ in paths:

        # Read in 
        df = pd.read_parquet(path_)

        # Determine metric cols
        metric_cols = [
            x for x in df.columns if
            any(p in y for p in pitcher_metrics) or
            any(b in y for b in batter_metrics)
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


def filter_dimensions(date):
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
    
    


if __name__ == "__main__":

    # Full dataset path
    years = ['2017']
    min_fill_thresh = 0.96
    
    top_pitcher_count = 15
    pitcher_metrics = ['BF', 'ER', 'ERA', 'HitsAllowed', 'Holds',
                       'SeasonLosses', 'SeasonWins', 'numberPitches',
                       'Outs', 'RunsAllowed', 'Strikes', 'SO']
    top_batter_count = 15
    batter_metrics = ['Assists', 'AB', 'BB', 'FO', 'Avg', 'H',
                      'HBP', 'HR', 'Doubles' 'GroundOuts', 'batterLob',
                      'OBP', 'OPS', 'R', 'RBI', 'SluggingPct',
                      'StrikeOuts', 'Triples']    

    # File list
    filelist_year = [
        CONFIG.get('paths').get('initial_featurespace')+fname
        for fname in os.listdir(
            CONFIG.get('paths').get('initial_featurespace')
        ) if any(y in fname for y in years)
    ]

    df_all_year = read_in_and_filter(filelist_year)
    
        

    # Index and Metrics
    idx_cols = [
        'home_team_win_pct_at_home', 'away_team_win_pct_at_away',
        'home_team_winner'
    ]
    target = 'home_team_winner'

    # Match Games
    df_matchups = match_games(df_all_year)
    
    # Filter Columns
    df_matchups = filter_dimensions(df_matchups)

    # Transformations
    df_matchups = dimension_fill_filter(
    
    
    
