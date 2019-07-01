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


def get_hist_boxscores(year):
    """
    """
    df_boxscores = []
    fnames = [CONFIG.get('paths').get('normalized')+dstr+
              "/boxscore.parquet"
              for dstr in os.listdir(
                      CONFIG.get('paths').get('normalized')
              )
              if os.path.isfile(
                      CONFIG.get('paths').get('normalized')+
                      dstr+"/boxscore.parquet"
              )
              and year in dstr]
    for fname in fnames:
        df = pd.read_parquet(fname)
        df = df.loc[:, [
            'gameId', 'date', 'away_team_flag', 'home_team_flag',
            'away_wins', 'away_loss', 'home_wins', 'home_loss',
        ]]
        df.loc[:, 'away_code'] = df['gameId'].apply(
            lambda x: x.split("_")[4][:3]
        )
        df.loc[:, 'home_code'] = df['gameId'].apply(
            lambda x: x.split("_")[5][:3]
        )
        df_boxscores.append(df)
    df_boxscores = pd.concat(
        objs=df_boxscores,
        axis=0
    )
    return df_boxscores


def get_prev_game_id(boxscore_hist):
    """
    """

    # Team Tables
    team_tables = []

    # Sort gameIds for each team
    for team in list(set(
            list(boxscore_hist['away_code']) +
            list(boxscore_hist['home_code'])
    )):
        # Combine Home and Away
        team_data = boxscore_hist.loc[
            boxscore_hist['gameId'].str.contains(team), :]
        team_data.sort_values(by=['gameId'], ascending=True, inplace=True)
        team_data.loc[:, 'prevGameId'] = team_data['gameId'].shift(1)
        team_data.loc[:, 'team_code'] = team
        team_tables.append(team_data[['gameId', 'team_code', 'prevGameId']])   
    team_tables = pd.concat(
        objs=team_tables,
        axis=0
    )
    return team_tables


def add_prev_game_ids(data, prev):
    """
    """
    data = pd.merge(
        data,
        prev,
        how='left',
        left_on=['gameId', 'home_code'],
        right_on=['gameId', 'team_code'],
        validate='1:1'
    )
    data.drop(labels=['team_code'], axis=1, inplace=True)
    data.rename(
        columns={'prevGameId': 'homePrevGameId'},
        inplace=True
    )
    data = pd.merge(
        data,
        prev,
        how='left',
        left_on=['gameId', 'away_code'],
        right_on=['gameId', 'team_code'],
        validate='1:1'
    )
    data.drop(labels=['team_code'], axis=1, inplace=True)
    data.rename(
        columns={'prevGameId': 'awayPrevGameId'},
        inplace=True
    )
    return data


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


def add_team_codes(data):
    """
    Parse out home and away from gameId
    """

    # Home
    data.loc[:, 'home_code'] = \
        data['gameId'].apply(
            lambda x: x.split("_")[5][:3]
        )
    data.loc[:, 'away_code'] = \
        data['gameId'].apply(
            lambda x: x.split("_")[4][:3]
        )
    return data


def starter_details_indicators(data):
    """
    """

    data.loc[:, 'home_starter_is_lhp'] = (
        data['home_starting_pitcher_throws'].str.lower() == 'l'
    ).astype(int)
    data.loc[:, 'away_starter_is_lhp'] = (
        data['away_starting_pitcher_throws'].str.lower() == 'l'
    ).astype(int)
    data.drop(
        labels=['home_starting_pitcher_throws',
                'away_starting_pitcher_throws'],
        axis=1,
        inplace=True
    )

    # Height
    data['home_starting_pitcher_height'].fillna("6-4", inplace=True)
    data['away_starting_pitcher_height'].fillna("6-4", inplace=True)
    data.loc[:, 'home_starting_pitcher_height'] = \
        data['home_starting_pitcher_height'].apply(
            lambda x: (
                (int(str(x).split("-")[0])*12) +
                int(str(x).split("-")[1])
            ) if "-" in str(x) else int(x)
        )
    data.loc[:, 'away_starting_pitcher_height'] = \
        data['away_starting_pitcher_height'].apply(
            lambda x: (
                (int(str(x).split("-")[0])*12) +
                int(str(x).split("-")[1])
            ) if "-" in str(x) else int(x)
        )
    
    # Age
    #print(data[['gameDate', 'home_starting_pitcher_dob',
    #            'away_starting_pitcher_dob']].head())
    #data.loc[:, 'home_starting_pitcher_age'] = (
    #    data['gameDate'] - 
    #    data['home_starting_pitcher_dob']
    #).dt.days / 365
    #data.loc[:, 'away_starting_pitcher_age'] = (
    #    data['gameDate'] -
    #    data['away_starting_pitcher_dob']
    #).dt.days / 365
    #data.drop(
    #    labels=['home_starting_pitcher_dob',
    #            'away_starting_pitcher_dob'],
    #    axis=1,
    #    inplace=True
    #)

    return data


def add_date(data):
    """
    Parse date from gameId 
    """

    # Parse
    data['gameDate'] = data['gameId'].apply(
        lambda d: str(d)[4:14]
    )
    data.loc[:, 'gameDate'] = data['gameDate'].astype(str)
    data.loc[:, 'gameDate'] = data['gameDate'].apply(
        lambda x: x.replace("_", "-")
    )
    data.loc[:, 'gameDate'] = \
        pd.to_datetime(data['gameDate'],
                       format="%Y-%m-%d")
    assert all(data['gameDate'].notnull())
    return data

    

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

    # ---------- ---------- ----------
    # Get Historic Boxscores
    df_box_hist = get_hist_boxscores(year)
    df_prev_game_ids = get_prev_game_id(df_box_hist)

    # ----------  ----------  ----------
    # Read in Line Score to get basis for each game played
    df_matchup_base = get_matchup_base_table(year)
    df_matchup_base = add_date(df_matchup_base)
    
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

    # Construct(ed) game-level | away | home
    df_matchup_base = df_matchup_base.loc[:, [
        'gameId', 'gameDate',
        'away_starting_pitcher', 'home_starting_pitcher'
    ]]
    
    df_matchup_base = add_starter_details(df_matchup_base)
    
    # ----------  ----------  ----------
    # Add Indicator Vars for Starter Details
    df_matchup_base = starter_details_indicators(df_matchup_base)
    
    # ----------  ----------  ----------
    # Add Team Prev Game ID
    df_matchup_base = add_team_codes(df_matchup_base)

    # ----------  ----------  ----------
    # Add Previous Game IDs
    df_matchup_base = add_prev_game_ids(df_matchup_base, df_prev_game_ids)
    df_matchup_base = df_matchup_base.loc[(
        (df_matchup_base['homePrevGameId'].notnull())
        &
        (df_matchup_base['awayPrevGameId'].notnull())
        &
        (df_matchup_base['away_starting_pitcher'].notnull())
        &
        (df_matchup_base['home_starting_pitcher'].notnull())
    ), :]
    df_matchup_base.to_csv(
        '/Users/peteraltamura/Desktop/df_matchup_base_hist_details.csv',
        index=False
    )
    
    # ----------  ----------  ----------
    # # # #
    # WIN / LOSS TARGET
    # # # #

    # ----------  ----------  ----------
    # Add Home starter trailing stats
    
    # Add Away Starter trailing stats
    
