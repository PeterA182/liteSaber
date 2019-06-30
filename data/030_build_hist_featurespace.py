import os
import gc
import sys
import pandas as pd
import datetime as dt
import numpy as np
import utilities as util
CONFIG = util.load_config()

"""

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

    # Add Date
    df.loc[:, 'gameDate'] = pd.to_datetime(
            df['gameDate'], infer_datetime_format=True
    )

    # cols
    cols = ['home_starting_pitcher', 'away_starting_pitcher']
    df = df.loc[:, ['gameId'] + cols].drop_duplicates(inplace=False)
    print("Starting Pitchers table shape")
    print(df.shape)
    return df


def add_starter_details(data):
    """
    """

    last_registries = [
        fname for fname in sorted(os.listdir(ref_dest))[-50:]
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
    registry.to_csv('/Users/peteraltamura/Desktop/registry_all.csv')
    registry.reset_index(drop=True, inplace=True)
    registry.drop_duplicates(
        subset=['first_name', 'last_name', 'team'],
        inplace=True
    )

    # Merge
    data = pd.merge(
        data,
        registry,
        how='left',
        left_on=['startingPitcherId'],
        right_on=['id'],
        validate='1:1',
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
                dd + "/" + "game_linescore_summary.parquet"
            ) for dd in os.listdir(
                CONFIG.get('paths').get('normalized')
            ) if str(year) in dd and
            os.path.isfile(CONFIG.get('paths').get('normalized') + \
                dd + "/" + "game_linescore_summary.parquet"
            )
        ],
        axis=0
    )
    return df_base    
    
    
    

if __name__ == "__main__":

    outpath = "/Users/peteraltamura/Desktop/"

    
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
    
