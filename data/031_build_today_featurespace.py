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


def add_series_game_indicator(df_current, d):
    """
    """
    
    # Read in boxscore from previous games in season
    paths = [
        CONFIG.get('paths').get('normalized') + \
        date_str + "/boxscore.parquet" for date_str in
        os.listdir(CONFIG.get('paths').get('normalized'))
    ]
    paths = [
        p for p in paths if (
            (str(d.year) in p.split("/")[4])
        ) and os.path.isfile(p)
    ]
    paths = [
        p for p in paths if (
            p.split("/")[4] != "year_{}month_{}day_{}".format(
                str(d.year),
                str(d.month).zfill(2),
                str(d.day).zfill(2)
            )
        )
    ]
    paths = [p for p in paths if os.path.exists(p)]
    df_boxscore = pd.concat(
        objs=[pd.read_parquet(p) for p in paths],
        axis=0
    )
    df_boxscore = add_team_codes(df_boxscore)
    df_boxscore = df_boxscore.loc[:, [
        'gameId', 'home_code', 'away_code',
    ]]
    curr_table = df_current.loc[:, [
        'gameId', 'home_code', 'away_code'
    ]]

    df_boxscore = pd.concat(
        objs=[df_boxscore, curr_table],
        axis=0
    )

    # Game Date
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
    df_boxscore = df_boxscore.loc[df_boxscore['gameDate'] == d, :]
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


def add_date(data):
    """
    Parse date from gameId 
    """

    # Parse
    data['gameDate'] = data['gameId'].apply(
        lambda d: str(d)[4:14]
    )
    data['gameDate'] = data['gameDate'].astype(str)
    data['gameDate'] = data['gameDate'].apply(
        lambda x: x.replace("_", "-")
    )
    data.loc[:, 'gameDate'] = \
        pd.to_datetime(data['gameDate'],
                       infer_datetime_format=True)
    assert all(data['gameDate'].notnull())
    return data


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
    return df_boxscores


def get_prev_game_id(boxscore_hist, curr_date):
    """
    THIS IS DIFFERENT FOR TODAY FEATURESPACE
    """

    # Team Tables
    team_tables = []

    # Add date to boxscore
    boxscore_hist = add_date(boxscore_hist)
    print(boxscore_hist[['gameDate']].shape)
    boxscore_hist = boxscore_hist.loc[
        boxscore_hist['gameDate'] < curr_date, :]
    print(boxscore_hist.shape)
    
    # Sort gameIds for each team
    for team in list(set(
            list(boxscore_hist['away_code']) +
            list(boxscore_hist['home_code'])
    )):

        # Combine Home and Away
        team_data = boxscore_hist.loc[
            boxscore_hist['gameId'].str.contains(team), :]
        max_date = max(team_data['gameDate'])
        team_data.sort_values(by=['gameDate'], ascending=True, inplace=True)
        team_data.loc[:, 'team_code'] = team
        team_data = team_data.loc[
            team_data['gameDate'] == max_date, :]
        team_data['rank'] = team_data.index + 1
        team_data = team_data.loc[
            team_data['rank'] == np.max(team_data['rank']), :]
        assert team_data.shape[0] <= 2
        if team_data.shape[0] > 1:
            team_data = team_data.tail(1)
        
        team_data.rename(columns={'gameId': 'prevGameId'}, inplace=True)
        team_data.reset_index(drop=True, inplace=True)
        
        team_tables.append(team_data[['team_code', 'prevGameId']])   
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
        left_on=['home_code'],
        right_on=['team_code'],
        validate='m:1'
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
        left_on=['away_code'],
        right_on=['team_code'],
        validate='m:1'
    )
    data.drop(labels=['team_code'], axis=1, inplace=True)
    data.rename(
        columns={'prevGameId': 'awayPrevGameId'},
        inplace=True
    )
    return data


def get_matchup_base_table(date):
    """
    """

    # Read in current year Summaries (will be cut down later)
    df_base = pd.read_parquet(
        CONFIG.get('paths').get('raw') + \
        date.strftime("year_%Ymonth_%mday_%d") +
        "/" + "probableStarters.parquet"
    )

    return df_base


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

    # TODO TODO data has not starters
    data = data.loc[(
        (data['home_starting_pitcher'].notnull())
        &
        (data['away_starting_pitcher'].notnull())
    ), :]
    data = pd.merge(
        data,
        registry,
        how='left',
        left_on=['home_starting_pitcher'],
        right_on=['id'],
        validate='1:1',
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
        validate='1:1',
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

    # Handed Throwing
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
                (int(x.split("-")[0])*12) +
                int(x.split("-")[1])
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


def get_starters_today(d):
    """
    """

    # Today review
    df_preview = pd.read_parquet(
        CONFIG.get('paths').get('raw') + \
        "year_{}month_{}day_{}/".format(
            str(d.year),
            str(d.month).zfill(2),
            str(d.day).zfill(2)
        ) + "probableStarters.parquet"
    )

    # Filter to both not null
    df_preview = df_preview.loc[(
        (df_preview['startingPitcherId_home'].notnull())
        &
        (df_preview['startingPitcherId_away'].notnull())
    ), :]
    df_preview = df_preview.loc[:, [
        'gameId', 'startingPitcherId_home', 'startingPitcherId_away'
    ]]
    df_preview.rename(
        columns={
            'startingPitcherId_home': 'home_starting_pitcher',
            'startingPitcherId_away': 'away_starting_pitcher'
        },
        inplace=True
    )
    return df_preview


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
    #data['home_starting_pitcher_id_rank'] -= 1
    #data['away_starting_pitcher_id_rank'] -= 1

    # Merge
    data.to_csv('/Users/peteraltamura/Desktop/data_pre_merge_1.csv', index=False)
    df_pitching.to_csv('/Users/peteraltamura/Desktop/df_pitching_pre_merge_1.csv', index=False)
    data = pd.merge(
        data,
        df_pitching,
        how='left',
        left_on=['away_starting_pitcher_id_rank', 'away_starting_pitcher_id'],
        right_on=['rank', 'pitcherId'],
        validate='1:1',
        suffixes=['', '_away']
    )
    data = pd.merge(
        data,
        df_pitching,
        how='left',
        left_on=['home_starting_pitcher_id_rank', 'home_starting_pitcher_id'],
        right_on=['rank', 'pitcherId'],
        validate='1:1',
        suffixes=['', '_home']
    )
    return data    


if __name__ == "__main__":


    # ----------  ----------  ----------
    # Parameters
    date = dt.datetime(year=2019, month=7, day=3)
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

    # Paths
    ref_dest = "/Volumes/Transcend/99_reference/"

    
    # ---------- ---------- ----------
    # Get Basic Matchup Table
    df_matchup_base = get_matchup_base_table(date=date)
    df_matchup_base = add_date(df_matchup_base)
    df_matchup_base = add_team_codes(df_matchup_base)
    df_matchup_base = add_series_game_indicator(df_matchup_base, date)

    # ---------- ---------- ----------
    # Get Historic Boxscores
    df_box_hist = get_hist_boxscores(str(date.year))
    df_box_hist.sort_values(by=['gameId'], ascending=True, inplace=True)    
    df_prev_game_ids = get_prev_game_id(df_box_hist, date)

    # Narrow to immediate dimensions
    starter_table = get_starters_today(date)

    starter_table = starter_table[[
        'gameId', 'home_starting_pitcher', 'away_starting_pitcher'
    ]]

    df_matchup_base = pd.merge(
        df_matchup_base,
        starter_table,
        how='left',
        on=['gameId'],
        validate='1:1'
    )
    
    # Narrow to immediate dimensions
    df_matchup_base = df_matchup_base.loc[:, [
        'gameId', 'gameDate',
        'home_starting_pitcher', 'away_starting_pitcher'
    ]]
    
    
    # Narrow down fields for matchup base table
    df_matchup_base = df_matchup_base.loc[:, [
        'gameId', 'gameDate',
        'away_starting_pitcher', 'home_starting_pitcher'
    ]]
    df_matchup_base = add_starter_details(df_matchup_base)

    # ----------  ----------  ----------
    # Add Indicator Vars for Starter Details
    df_matchup_base = starter_details_indicators(df_matchup_base)

    # ----------  ----------  ----------
    # Add Team Codes
    df_matchup_base = add_team_codes(df_matchup_base)

    # ----------  ----------  ----------
    # Add Previous Game IDs for Merging
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
    # THIS IS WHERE WIN / LOSS WOULD BE ADDED -- TARGET
    # # # #
    
    df_matchup_base.rename(
        columns={'home_starting_pitcher': 'home_starting_pitcher_id',
                 'away_starting_pitcher': 'away_starting_pitcher_id'},
        inplace=True
    )
    
    # ----------  ----------  ----------
    # Add starter trailing stats
    df_matchup_base = add_starter_stats(data=df_matchup_base, years=[str(date.year)])
    df_matchup_base.to_csv(
        '/Users/peteraltamura/Desktop/df_matchup_base_tomorrow_details.csv',
        index=False
    )


    
