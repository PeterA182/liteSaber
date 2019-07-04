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


def add_series_game_indicator(df_boxscore):
    """
    """
    df_boxscore.loc[:, 'gameDate'] = \
        df_boxscore['gameId'].apply(
            lambda x: x.split("_")
        )
    df_boxscore.loc[:, 'gameDate'] = \
        df_boxscore['gameDate'].apply(
            lambda d: dt.datetime(
                year=int(d[1]),
                month=int(d[2]),
                day=int(d[3])
            )
        )

    # Add Date
    df_boxscore.loc[:, 'gameDate'] = pd.to_datetime(
            df_boxscore['gameDate'], infer_datetime_format=True
    )
    
    # Do for all home teams
    df_boxscore.sort_values(
        by=['home_code', 'gameDate'],
        ascending=True,
        inplace=True
    )
    df_boxscore.loc[:, 'series_game_1_flag'] = 0
    df_boxscore.loc[(
        df_boxscore['away_code'] !=
        df_boxscore['away_code'].shift(1)
    ), 'series_game_1_flag'] = 1
    df_boxscore.loc[:, 'series_game_2_flag'] = 0
    df_boxscore.loc[(
        (df_boxscore['away_code'] == df_boxscore['away_code'].shift(1))
        &
        (df_boxscore['series_game_1_flag'].shift(1) == 1)
    ), 'series_game_2_flag'] = 1
    if any(
        (df_boxscore['series_game_2_flag'].shift(1) == 1)
        &
        (df_boxscore['away_code']==df_boxscore['away_code'].shift(1))
        &
        (df_boxscore['home_code']==df_boxscore['home_code'].shift(1))
    ):
        df_boxscore.loc[:, 'series_game_3_flag'] = 0
        df_boxscore.loc[(
            (df_boxscore['series_game_2_flag'].shift(1) == 1)
            &
            (df_boxscore['away_code']==df_boxscore['away_code'].shift(1))
            &
            (df_boxscore['home_code']==df_boxscore['home_code'].shift(1))
        ), 'series_game_3_flag'] = 1
        if any(
                (df_boxscore['series_game_3_flag'].shift(1) == 1)
                &
                (df_boxscore['series_game_1_flag'] == 0)
        ):
            df_boxscore.loc[:, 'series_game_4_flag'] = 0
            df_boxscore.loc[(
                (df_boxscore['series_game_3_flag'].shift(1) == 1)
                &
                (df_boxscore['series_game_1_flag'] == 0)
            ), 'series_game_4_flag'] = 1
    return df_boxscore


def add_disposition_flags(df_boxscore):
    """
    """
    if 'series_game_1_flag' not in df_boxscore.columns:
        df_boxscore = add_series_game_indicator(df_boxscore)
    df_boxscore.loc[:, 'gameDate'] = \
        df_boxscore['gameId'].apply(
            lambda x: x.split("_")
        )
    df_boxscore.loc[:, 'gameDate'] = \
        df_boxscore['gameDate'].apply(
            lambda d: dt.datetime(
                year=int(d[1]),
                month=int(d[2]),
                day=int(d[3])
            )
        )

    # Add Date
    df_boxscore.loc[:, 'gameDate'] = pd.to_datetime(
            df_boxscore['gameDate'], infer_datetime_format=True
    )
    
    # Do for all home teams
    df_boxscore.sort_values(
        by=['home_code', 'gameDate'],
        ascending=True,
        inplace=True
    )
    init_row = df_boxscore.shape[0]
    
    # home_last_series_home
    results_home = []
    results_away = []
    teams = list(set(
        list(df_boxscore['away_code']) + \
        list(df_boxscore['home_code'])
    ))
    
    for team in teams:
        curr_box = df_boxscore.loc[(
            (df_boxscore['home_code'] == team)
            |
            (df_boxscore['away_code'] == team)
        ), :]
        curr_box.sort_values(
            by=['gameDate'],
            ascending=True,
            inplace=True
        )

        # Home team last series home
        curr_box.loc[:, 'home_team_last_series_home'] = 0
        curr_box.loc[(
            (curr_box['series_game_1_flag'] == 1)
            &
            (curr_box['home_code'] == curr_box['home_code'].shift(1))
            &
            (curr_box['home_code'] == team)
        ), 'home_team_last_series_home'] = 1
    
        # Away team last series away
        curr_box.loc[:, 'away_team_last_series_away'] = 0
        curr_box.loc[(
            (curr_box['series_game_1_flag'] == 1)
            &
            (curr_box['away_code'] == curr_box['away_code'].shift(1))
            &
            (curr_box['away_code'] == team)
        ), 'away_team_last_series_away'] = 1

        # Fill forward 1
        curr_box.reset_index(drop=True, inplace=True)
        curr_box.loc[(
            (curr_box['away_code'] == curr_box['away_code'].shift(1))
            &
            (curr_box['home_code'] == curr_box['home_code'].shift(1))
        ), 'home_team_last_series_home'] = curr_box['home_team_last_series_home'].shift(1)
        curr_box.loc[(
            (curr_box['away_code'] == curr_box['away_code'].shift(1))
            &
            (curr_box['home_code'] == curr_box['home_code'].shift(1))
        ), 'away_team_last_series_away'] = curr_box['away_team_last_series_away'].shift(1)

        # Fill forward 2
        curr_box.reset_index(drop=True, inplace=True)
        curr_box.loc[(
            (curr_box['away_code'] == curr_box['away_code'].shift(1))
            &
            (curr_box['home_code'] == curr_box['home_code'].shift(1))
        ), 'home_team_last_series_home'] = curr_box['home_team_last_series_home'].shift(1)
        curr_box.loc[(
            (curr_box['away_code'] == curr_box['away_code'].shift(1))
            &
            (curr_box['home_code'] == curr_box['home_code'].shift(1))
        ), 'away_team_last_series_away'] = curr_box['away_team_last_series_away'].shift(1)

        # Combine
        add1 = curr_box.loc[curr_box['home_code'] == team, :][[
            'gameId', 'home_code',
            'home_team_last_series_home'
        ]]
        add2 = curr_box.loc[curr_box['away_code'] == team, :][[
            'gameId', 'away_code',
            'away_team_last_series_away'
        ]]
        results_home.append(add1)
        results_away.append(add2)
        
    # Concatenate Home and Away in prep for merge
    results_home = pd.concat(
        objs=results_home,
        axis=0
    )
    results_away = pd.concat(
        objs=results_away,
        axis=0
    )

    # Merge
    df_boxscore = pd.merge(
        df_boxscore,
        results_home,
        how='left',
        on=['gameId', 'home_code'],
        validate='1:1'
    )
    df_boxscore = pd.merge(
        df_boxscore,
        results_away,
        how='left',
        on=['gameId', 'away_code'],
        validate='1:1'
    )
    # The above should have duplicated the entire table
    #   (by iterating over team)
    assert df_boxscore.shape[0] == init_row
    return df_boxscore


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

    # Add game number in series indicators
    df_boxscores = add_series_game_indicator(df_boxscores)
    
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


def add_target(data):
    """
    """

    # Read in all hist line scores with current year in path
    df_ls = pd.concat(
        objs=[
            pd.read_parquet(
                CONFIG.get('paths').get('raw') + \
                dd + "/" + "linescores.parquet"
            ) for dd in os.listdir(
                CONFIG.get('paths').get('raw')
            ) if str(year) in dd and
            os.path.isfile(CONFIG.get('paths').get('raw') + \
                dd + "/" + "linescores.parquet"
            )
        ],
        axis=0
    )
    df_ls = df_ls.loc[:, ['gameId', 'home_team_runs', 'away_team_runs']]
    for col in ['home_team_runs', 'away_team_runs']:
        df_ls.loc[:, col] = df_ls[col].astype(float)
    df_ls['home_team_winner'] = (df_ls['home_team_runs'] >
                                 df_ls['away_team_runs']).astype(int)
    data = pd.merge(
        data,
        df_ls[['gameId', 'home_team_winner']],
        how='left',
        on=['gameId'],
        validate='1:1'
    )
    return data


def add_starter_stats(data, years):
    """
    """

    # Get full paths to all files
    paths = [
        CONFIG.get('paths').get('pitcher_saber')+gid+"/pitcher_saber.csv"
        for gid in os.listdir(CONFIG.get('paths').get('pitcher_saber'))
        if any(y in gid for y in years)
    ]
    # Concatenate all paths created
    df_pitching = pd.concat(
        objs=[pd.read_csv(fpath) for fpath in paths if fpath[-4:] == ".csv"],
        axis=0
    )
    df_pitching.loc[:, 'pitcherId'] = df_pitching['pitcherId'].astype(str)
    # Sort ascending pitcherId and date
    df_pitching.sort_values(
        by=['pitcherId', 'gameDate'],
        ascending=True,
        inplace=True
    )
    df_pitching.reset_index(drop=True, inplace=True)
    df_pitching.loc[:, 'rankItem'] = 1
    df_pitching['rank'] = df_pitching.groupby('pitcherId')['rankItem'].cumcount()

    # Get most recently available stats per pitcher
    # Rank pitcherId appearances in data
    data.sort_values(
        by=['home_starting_pitcher_id', 'gameDate'],
        ascending=True,
        inplace=True
    )
    data.reset_index(drop=True, inplace=True)
    data.loc[:, 'rankItem'] = 1
    data.loc[:, 'home_starting_pitcher_id_rank'] = \
        data.groupby('home_starting_pitcher_id')['rankItem'].cumcount()
    data.sort_values(
        by=['away_starting_pitcher_id', 'gameDate'],
        ascending=True,
        inplace=True
    )
    data.reset_index(drop=True, inplace=True)
    data.loc[:, 'rankItem'] = 1
    data['away_starting_pitcher_id_rank'] = \
        data.groupby('away_starting_pitcher_id')['rankItem'].cumcount()
    # Decrease Perf Rank by 1
    data['home_starting_pitcher_id_rank'] -= 1
    data['away_starting_pitcher_id_rank'] -= 1

    # ----------
    # Merge Away
    df_pitching_away = df_pitching.rename(
        columns={k: k+"_AWAY" for k in df_pitching.columns},
        inplace=False
    )
    data = pd.merge(
        data,
        df_pitching_away,
        how='left',
        left_on=['away_starting_pitcher_id_rank', 'away_starting_pitcher_id'],
        right_on=['rank_AWAY', 'pitcherId_AWAY'],
        validate='1:1'
    )
    del df_pitching_away
    gc.collect()

    # ----------
    # Merge Home
    df_pitching_home = df_pitching.rename(
        columns={k: k+"_HOME" for k in df_pitching.columns},
        inplace=False
    )
    data = pd.merge(
        data,
        df_pitching_home,
        how='left',
        left_on=['home_starting_pitcher_id_rank', 'home_starting_pitcher_id'],
        right_on=['rank_HOME', 'pitcherId_HOME'],
        validate='1:1'
    )
    del df_pitching_home
    gc.collect()
    return data


def add_team_batting_stats(df, years, batter_metrics):
    """
    """
    gids = list(set(df['gameId']))
    bat_saber_paths = [
        CONFIG.get('paths').get('batter_saber') + gid + \
        "/batter_saber_team.parquet" for gid in
        os.listdir(CONFIG.get('paths').get('batter_saber'))
    ]
    curr_gids = list(set(
        list(df['homePrevGameId']) +
        list(df['awayPrevGameId'])
    ))
    bat_saber_paths = [
        x for x in bat_saber_paths if any(
            gid in x for gid in curr_gids
        )
    ]
    batter_saber = pd.concat(
        objs=[pd.read_parquet(path) for path in bat_saber_paths],
        axis=0
    )
    print(batter_saber.shape)
    print("batter saber shape above")

    # Get top 9 by AB
    batter_saber['game_id_team'] = (
        batter_saber['gameId'] + batter_saber['team']
    )
    batter_saber.sort_values(by=['game_id_team', 'woba_trail6'],
                             ascending=False,
                             inplace=True)
    batter_saber['rank'] = batter_saber.groupby('game_id_team')\
        ['batterId'].cumcount()
    batter_saber = batter_saber.loc[batter_saber['rank'] <= 9, :]
    batter_saber.loc[batter_saber['rank'] < 5, 'batter_group'] = 'high'
    batter_saber.loc[batter_saber['rank'] >= 5, 'batter_group'] = 'low'

    # Aggregate
    batter_saber = batter_saber.groupby(
        by=['gameId', 'team', 'batter_group'],
        as_index=False
    ).agg({k: 'mean' for k in batter_metrics})
    
    batter_saber = batter_saber.pivot_table(
        index=['gameId', 'team'],
        columns=['batter_group'],
        values=[k for k in batter_metrics],
        aggfunc='mean'
    )
    batter_saber.reset_index(inplace=True)
    batter_saber.columns = [
        x[0] if x[1] == '' else x[0]+"_"+x[1]
        for x in batter_saber.columns
    ]

    # ----------
    # Merge Home
    batter_saber_home = batter_saber.rename(
        columns={
            k: k+"_HOME" for k in list(batter_saber.columns)
        },
        inplace=False
    )
    df = pd.merge(
        df,
        batter_saber_home,
        how='left',
        left_on=['homePrevGameId', 'home_code'],
        right_on=['gameId_HOME', 'team_HOME'],
        validate='1:1'
    )
    #df.drop(labels=['gameId_HOME', 'team_HOME'],
    #        axis=1,
    #        inplace=True)
    del batter_saber_home
    gc.collect()

    # ----------
    # Merge Away
    batter_saber_away = batter_saber.rename(
        columns={
            k: k+"_AWAY" for k in list(batter_saber.columns)
        },
         inplace=False
    )
    df = pd.merge(
        df,
        batter_saber_away,
        how='left',
        left_on=['awayPrevGameId', 'away_code'],
        right_on=['gameId_AWAY', 'team_AWAY'],
        validate='1:1'
    )
    #df.drop(labels=['gameId_AWAY', 'team_AWAY'],
    #        axis=1,
    #        inplace=True)
    del batter_saber_away
    gc.collect()
    return df


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
    #batter_metrics = [
    #    'Assists', 'AB', 'BB', 'FO', 'Avg', 'H',
    #    'HBP', 'HR', 'Doubles' 'GroundOuts', 'batterLob',
    #    'OBP', 'OPS', 'R', 'RBI', 'SluggingPct',
    #    'StrikeOuts', 'Triples'
    #]
    batter_metrics = [
        'batterWalkPercentage_trail3', 'batterWalkPercentage_trail6',
        'batterKPercentage_trail3', 'batterKPercentage_trail6',
        'batterISO_trail3', 'batterISO_trail6', 'batterBABIP_trail3',
        'batterBABIP_trail6', 'woba_trail3', 'woba_trail6',
        'ab_trail3', 'ab_trail6'
    ]
    featurespace = []

    # ---------- ---------- ----------
    # Get Historic Boxscores
    df_box_hist = get_hist_boxscores(year)
    featurespace.extend(['series_game_1_flag', 'series_game_2_flag',
                         'series_game_3_flag', 'series_game_4_flag'])

    # ---------- ---------- ----------
    # Add last series disposition
    df_box_hist = add_disposition_flags(df_box_hist)
    featurespace.extend([
        'away_team_last_series_away', 'home_team_last_series_home'
    ])

    # Add Previous Game Id
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
    
    # ----------  ----------  ----------
    # # # #
    # WIN / LOSS TARGET
    # # # #
    df_matchup_base = add_target(df_matchup_base)
    df_matchup_base.rename(
        columns={'home_starting_pitcher': 'home_starting_pitcher_id',
                 'away_starting_pitcher': 'away_starting_pitcher_id'},
        inplace=True
    )

    # ----------  ----------  ----------
    # Add Home starter trailing stats and
    #     add Away Starter trailing stats
    df_matchup_base = add_starter_stats(data=df_matchup_base, years=[year])

    # Indicators for game number of series
    df_matchup_base = add_series_game_indicator(df_matchup_base)

    # Add Batting Stats
    df_matchup_base = add_team_batting_stats(df=df_matchup_base,
                                              years=[year],
                                              batter_metrics=batter_metrics)

    # Save out
    fname = "{}_{}_hist_features.parquet".format(
        year,
        (dt.datetime.today()-dt.timedelta(days=1)).strftime("%Y_%m_%d")
    )
    try:
        df_matchup_base.to_parquet(
            CONFIG.get('paths').get('full_featurespaces') + fname
        )
    except:
        pass
    df_matchup_base.to_csv(
        CONFIG.get('paths').get('full_featurespaces') + \
        fname.replace(".parquet", ".csv"),
        index=False
    )
