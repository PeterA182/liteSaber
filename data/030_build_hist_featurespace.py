import os
import gc
import sys
import pandas as pd
import datetime as dt
import numpy as np
import utilities as util
CONFIG = util.load_config()

"""
GAMES ARE WIDE 

|| gameID || home_starting_pitcher || away_starting_pitcher || ... ||

"""


def get_starters():
    """
    """

    # Read in
    # Get all paths for year
    innings_paths = [
        CONFIG.get('paths').get('normalized') + date_str + "/innings.parquet"
        for date_str in os.listdir(CONFIG.get('paths').get('normalized'))
    ]
    inning_paths = [x for x in innings_paths if os.path.isfile(x)]
    
    # Process and append batting paths
    df = pd.concat(
            objs=[pd.read_parquet(i_path) for i_path in inning_paths],
            axis=0
    )
    df = df.loc[df['gameId'].notnull(), :]

    df.loc[:, 'gameDate'] = \
        df['gameId'].apply(
            lambda x: x.split("_")
        )
    df.loc[:, 'gameDate'] = df['gameDate'].apply(
        lambda d: dt.datetime(
            year=int(d[1]),
            month=int(d[2]),
            day=int(d[3])
        )
    )

    # Add Date
    df.loc[:, 'gameDate'] = pd.to_datetime(
            df['gameDate'], infer_datetime_format=True
    )

    # cols
    cols = ['home_starting_pitcher', 'away_starting_pitcher']
    df = df.loc[:, ['gameId'] + cols].drop_duplicates(inplace=False)
    return df


def add_starter_details(data):
    """
    """

    last_registries = [
        fname for fname in sorted(os.listdir(ref_dest))[-100:]
    ]
    registry = pd.concat(
        objs=[
            pd.read_parquet(ref_dest + fname) for fname in last_registries
        ],
        axis=0
    )
    for col in ['first_name', 'last_name']:
        registry.loc[:, col] = registry[col].astype(str)
        registry.loc[:, col] = \
            registry[col].apply(lambda x: x.lower().strip())
    registry.reset_index(drop=True, inplace=True)
    registry.drop_duplicates(
        subset=['id'],
        inplace=True
    )
    registry = registry.loc[:, [
        'id', 'height', 'throws', 'weight', 'dob'
    ]]

    # Merge for Home
    data = pd.merge(
        data,
        registry,
        how='left',
        left_on=['home_starting_pitcher'],
        right_on=['id'],
        validate='m:1',
    )
    data.drop(labels=['id'], axis=1, inplace=True)
    data.rename(
        columns={'height': 'home_starting_pitcher_height',
                 'throws': 'home_starting_pitcher_throws',
                 'weight': 'home_starting_pitcher_weight',
                 'dob': 'home_starting_pitcher_dob'},
        inplace=True
    )

    # Merge for Away
    data = pd.merge(
        data,
        registry,
        how='left',
        left_on=['away_starting_pitcher'],
        right_on=['id'],
        validate='m:1',
    )
    data.drop(labels=['id'], axis=1, inplace=True)
    data.rename(
        columns={'height': 'away_starting_pitcher_height',
                 'throws': 'away_starting_pitcher_throws',
                 'weight': 'away_starting_pitcher_weight',
                 'dob': 'away_starting_pitcher_dob'},
        inplace=True
    )
    return data


def get_matchup_base_table(year):
    """
    """

    # Read in current year Summaries (will be cut down later)
    df_base = pd.concat(
        objs=[
            pd.read_parquet(
                CONFIG.get('paths').get('normalized') + \
                dd + "/" + "boxscore.parquet"
            ) for dd in os.listdir(
                CONFIG.get('paths').get('normalized')
            ) if str(year) in dd and
            os.path.isfile(CONFIG.get('paths').get('normalized') + \
                dd + "/" + "boxscore.parquet"
            )
        ],
        axis=0
    )
    return df_base    
    
    
    

if __name__ == "__main__":

    outpath = "/Users/peteraltamura/Desktop/"
    ref_dest = "/Volumes/Transcend/99_reference/"

    
    # ----------  ----------  ----------
    # Parameters
    year = '2019'
    bullpen_top_pitcher_count = 6
    pitcher_metrics = [
        'BF', 'ER', 'ERA', 'HitsAllowed', 'Holds',
        'SeasonLosses', 'SeasonWins', 'numberPitches',
        'Outs', 'RunsAllowed', 'Strikes', 'SO'
    ]
    top_batter_count = 12
    batter_metrics = [
        'Assists', 'AB', 'BB', 'FO', 'Avg', 'H',
        'HBP', 'HR', 'Doubles' 'GroundOuts', 'batterLob',
        'OBP', 'OPS', 'R', 'RBI', 'SluggingPct',
        'StrikeOuts', 'Triples'
    ]

    # ----------  ----------  ----------
    # Read in Line Score to get basis for each game played
    df_matchup_base = get_matchup_base_table(year)

    # Narrow to immediate dimensions
    starter_table = get_starters()
    starter_table = starter_table.loc[:, [
        'gameId', 'home_starting_pitcher', 'away_starting_pitcher'
    ]]
    df_matchup_base = pd.merge(
        df_matchup_base,
        starter_table,
        how='left',
        on='gameId',
        validate='1:1'
    )
    print(df_matchup_base.columns)

    # Construct(ed) game-level | away | home
    df_matchup_base = df_matchup_base.loc[:, [
        'gameId', 'away_starting_pitcher', 'home_starting_pitcher'
    ]]
    df_matchup_base.to_csv('/Users/peteraltamura/Desktop/df_matchup_base_hist.csv', index=False)
    df_matchup_base = add_starter_details(df_matchup_base)
    df_matchup_base.to_csv('/Users/peteraltamura/Desktop/df_matchup_base_hist_details.csv', index=False)
