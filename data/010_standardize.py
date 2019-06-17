import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import inspect
import utilities as util
from pre_process_library import rename as names
from pre_process_library import additions as add
CONFIG = util.load_config()


def process_date_batting(data):
    """
    """

    # Rename
    data = names.rename_table(data, tablename='batting')
    data = names.remap_dtypes(data, tablename='batting')

    # Add Teams
    data = add.add_team(data, path, 'batter')

    # Add Date
    data = add.add_game_date(data, path)

    return data


def process_date_pitching(data):
    """
    """

    # Rename Pitching
    data = names.rename_table(data, tablename="pitching")
    data = names.remap_dtypes(data, tablename='pitching')

    # Add Teams
    data = add.add_team(data, path, 'pitcher')

    # Add Date
    data = add.add_game_date(data, path)

    return data


def process_date_boxscore(data, tablename='boxscore'):
    """
    """

    # Rename Boxscore
    data = names.rename_table(data, tablename='boxscore')
    return data


def process_date_innings(data, tablename='innings'):
    """
    """

    # Rename innings
    #data = names.rename_table(data, tablename='innings')
    data = add.add_game_date(data, path)
    data = add.add_inning_half(data)
    print("inning hald done")
    data = add.add_starting_pitcher_flag(data)
    print('still good')
    data = add.add_team(data, path, 'inning')
    return data


def process_date_games(path):
    """
    Script for applying processing methods to one date's worth of games
    
    PARAMETERS
    ----------
    path_: str
        path to date directory with game tables within
    
    returns:
        None
    """

    # Get Files
    files_ = os.listdir(path)

    # Create desination dir
    if not os.path.exists(
            CONFIG.get('paths').get('normalized') + \
            path.split("/")[-2]+"/"
    ):
        os.makedirs(CONFIG.get('paths').get('normalized') + \
                    path.split("/")[-2]+"/")
    
    # Process batting
    df = pd.read_parquet(path+"batting.parquet")
    df = process_date_batting(df)
    df.to_parquet(
        CONFIG.get('paths').get('normalized') + \
        path.split("/")[-2] + "/" + \
        "batting.parquet"
    )   
    print('batting done')
    # Process Pitching
    df = pd.read_parquet(path+"pitching.parquet")
    df = process_date_pitching(df)
    df.to_parquet(
        CONFIG.get('paths').get('normalized') + \
        path.split("/")[-2] + "/" + \
        "pitching.parquet"
    )
    print('pitching done')
    # Process Boxscore
    df = pd.read_parquet(path+"boxscore.parquet")
    df = process_date_boxscore(df)
    df.to_parquet(
        CONFIG.get('paths').get('normalized') + \
        path.split("/")[-2] + "/" + \
        "boxscore.parquet"
    )
    print('boxscore done')
    # Process Innings
    df = pd.read_parquet(path+"innings.parquet")
    df = process_date_innings(df)
    df.to_parquet(
        CONFIG.get('paths').get('normalized') + \
        path.split("/")[-2] + "/" + \
        "innings.parquet"
    )
    print('innings done')
    if 'day_01' in path:
        df.to_csv(
            CONFIG.get('paths').get('normalized') + \
            path.split("/")[-2] + "/" + \
            "innings.csv",
            index=False
        )

    # Process Game Linescore Summary
    df = pd.read_csv(path+"game_linescore_summary.csv", dtype=str)
    for col in ['home_win', 'home_loss', 'away_win', 'away_loss']:
        df.loc[:, col] = df[col].astype(float)
    df.to_parquet(
        CONFIG.get('paths').get('normalized') + \
        path.split("/")[-2] + "/" + \
        "game_linescore_summary.parquet"
    )
    if 'day_01' in path:
        df.to_csv(
            CONFIG.get('paths').get('normalized') + \
            path.split("/")[2] + "/" + \
            "game_linescore_summary.csv",
            index=False
        )
    
    

if __name__ == "__main__":

    # Run Log
    min_date = dt.datetime(year=2018, month=3, day=31)
    max_date = dt.datetime(year=2018, month=4, day=30)

    # Iterate over years
    years = [y for y in np.arange(min_date.year, max_date.year+1, 1)]
    dates = [min_date+dt.timedelta(days=i)
             for i in range((max_date-min_date).days+1)]

    # Estalish path
    for dd in dates:
        path = CONFIG.get('paths').get('raw')
        dd = dd.strftime('year_%Ymonth_%mday_%d/')
        path += dd
        
        print(dd)
        process_date_games(path)
        ##except:
        #    print("Failed")
        #    continue
