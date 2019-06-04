import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
from pre_process_library import batting as bats
import utilities as util
CONFIG = util.load_config()


def master_from_boxscore(paths):
    """
    """

    # Concatenate
    df_boxscores = pd.concat(
        objs=[pd.read_parquet(path_) for path_ in paths],
        axis=0
    )

    # Base return table
    df_return = df_boxscores.loc[:, [
        'gameId', 'away_team_flag', 'home_team_flag', 'date'
    ]]

    # Fill In away / home indicators
    for team in list(set(
        list(pd.Series.unique(df_return.away_team_flag)) +
        list(pd.Series.unique(df_return.home_team_flag))
    )):
        df_return.loc[:, '{}_is_away'.format(team)] = (
            df_return['away_team_flag'] == team
        ).astype(int)
        df_return.loc[:, '{}_is_home'.format(team)] = (
            df_return['home_team_flag'] == team
        ).astype(int)

    # Day of Week
    df_return.loc[:, 'date'] = pd.to_datetime(df_return['date'])
    dow = ['m', 't', 'w', 'r', 'f', 'st', 'sn']
    for d in dow:
        df_return.loc[:, '{}_is_day'.format(d)] = (
            df_return['date'].apply(
                lambda x: x.weekday()
            ) == dow.index(d)
        )
    return df_return


def master_pitching_from_innings(paths):
    """
    """

    # Concatenate
    df_innings = pd.concat(
        objs=[pd.read_parquet(path_) for path_ in paths],
        axis=0
    )

    # Base return table
    df_return = df_innings.loc[:, [
        'gameId', 'home_starting_pitcher', 'away_starting_pitcher'
    ]].drop_duplicates(inplace=False)
    assert df_return.shape[0] == pd.Series.nunique(df_return['gameId'])
    print(df_innings.shape)
    print(len(paths))
    



if __name__ == "__main__":

    # -----------------
    # Years
    min_year = 2018
    max_year = 2018
    if min_year == max_year:
        years = [min_year]
    else:
        years = [min_year+i for i in range(max_year+1)]

    # Process
    for yr in years:

        # Gets all paths for year
        year_paths = [
            CONFIG.get('paths').get('normalized')+foldername for foldername in [
                x for x in os.listdir(CONFIG.get('paths').get('normalized'))
                if str(yr) in x
            ]
        ]

        # Assemble lists of full paths for each normalized table category
        full_boxscore_paths = [p + "/boxscore.parquet" for p in year_paths]
        full_boxscore_paths = [p for p in full_boxscore_paths if os.path.isfile(p)]

        # Assemble master table from year's boxscores
        df_master_table = master_from_boxscore(full_boxscore_paths)
        
        # Assemble master table from innings
        df_master_table_innings = master_pitching_from_innings(full_boxscore_paths)
        
                            
    
    











