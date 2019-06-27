import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import inspect
import utilities as util
import multiprocessing as mp
from pre_process_library import rename as names
from pre_process_library import additions as add
CONFIG = util.load_config()


def process_date_batting(data, path):
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


def process_date_pitching(data, path):
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


def process_date_boxscore(data, path, tablename='boxscore'):
    """
    """

    # Rename Boxscore
    data = names.rename_table(data, tablename='boxscore')
    return data


def process_date_innings(data, path, tablename='innings'):
    """
    """

    # Rename innings
    #data = names.rename_table(data, tablename='innings')
    data = add.add_game_date(data, path)
    data = add.add_inning_half(data)
    data = add.add_starting_pitcher_flag(data)
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
    path = path

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
    df = process_date_batting(df, path)
    df.to_parquet(
        CONFIG.get('paths').get('normalized') + \
        path.split("/")[-2] + "/" + \
        "batting.parquet"
    )   
    print('    batting done')
    # Process Pitching
    df = pd.read_parquet(path+"pitching.parquet")
    df = process_date_pitching(df, path)
    df.to_parquet(
        CONFIG.get('paths').get('normalized') + \
        path.split("/")[-2] + "/" + \
        "pitching.parquet"
    )
    print('    pitching done')
    # Process Boxscore
    df = pd.read_parquet(path+"boxscore.parquet")
    df = process_date_boxscore(df, path)
    df.to_parquet(
        CONFIG.get('paths').get('normalized') + \
        path.split("/")[-2] + "/" + \
        "boxscore.parquet"
    )
    print('    boxscore done')
    # Process Innings
    df = pd.read_parquet(path+"innings.parquet")
    df = process_date_innings(df, path)
    df.to_parquet(
        CONFIG.get('paths').get('normalized') + \
        path.split("/")[-2] + "/" + \
        "innings.parquet"
    )
    if 'day_01' in path:
        df.to_csv(
            CONFIG.get('paths').get('normalized') + \
            path.split("/")[-2] + "/" + \
            "innings.csv",
            index=False
        )
    print('    innings done')
    # Save Starters
    df.loc[:, [
        'gameId', 'inning_home_team', 'inning_away_team',
        'home_starting_pitcher', 'away_starting_pitcher'
    ]].to_parquet(
        CONFIG.get('paths').get('normalized') + \
        path.split("/")[-2] + "/" + \
        "starters.parquet"
    )
    print("    starters done")
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
            path.split("/")[-2] + "/" + \
            "game_linescore_summary.csv",
            index=False
        )
    print("    linescore done")
    
    

if __name__ == "__main__":

    # Run Log
    min_date = dt.datetime(year=2018, month=3, day=31)
    max_date = dt.datetime(year=2018, month=4, day=30)

    # Iterate over years
    years = [y for y in np.arange(min_date.year, max_date.year+1, 1)]
    dates = [min_date+dt.timedelta(days=i)
             for i in range((max_date-min_date).days+1)]

    # Estalish path
    dates = [CONFIG.get('paths').get('raw') +
             dd.strftime('year_%Ymonth_%mday_%d/') for dd in dates]
    proc_ = mp.cpu_count()
    POOL = mp.Pool(proc_)
    r = POOL.map(process_date_games, dates)
    POOL.close()
    POOL.join()
#    for dd in dates:
#        path = CONFIG.get('paths').get('raw')
#        dd = dd.strftime('year_%Ymonth_%mday_%d/')
#        path += dd
        
#        print(dd)
#        process_date_games(path)
