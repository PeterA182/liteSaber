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
    return df_boxscores


def get_prev_game_id(boxscore_hist, curr_date):
    """
    THIS IS DIFFERENT FOR TODAY FEATURESPACE
    """

    # Team Tables
    team_tables = []

    # Add date to boxscore
    boxscore_hist = add_date(boxscore_hist)
    boxscore_hist.to_csv(
        '/Users/peteraltamura/Desktop/boxhist.csv',
        index=False
    )

    # Sort gameIds for each team
    for team in list(set(
            list(boxscore_hist['away_code']) +
            list(boxscore_hist['home_code'])
    )):
    #for team in ['sfn']:
        print(team)
        # Combine Home and Away
        team_data = boxscore_hist.loc[
            boxscore_hist['gameId'].str.contains(team), :]
        team_data.to_csv('/Users/peteraltamura/Desktop/sfn_box_missing.csv',
                         index=False)
        max_date = max(team_data['gameDate'])
        team_data.sort_values(by=['gameDate'], ascending=True, inplace=True)
        team_data.loc[:, 'team_code'] = team
        team_data = team_data.loc[
            team_data['gameDate'] == max_date, :]
        team_data['rank'] = team_data.index + 1
        team_data = team_data.loc[
            team_data['rank'] == np.max(team_data['rank']), :]
        assert team_data.shape[0] == 1
        
        team_data.rename(columns={'gameId': 'prevGameId'}, inplace=True)
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
        left_on=['away_code'],
        right_on=['team_code'],
        validate='1:1'
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

    # Merge for Home
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

    # Age
    #data['home_starting_pitcher_dob'] = \
    #    pd.to_datetime(data['home_starting_pitcher_dob'])
    #data['away_starting_pitcher_dob'] = \
    #    pd.to_datetime(data['away_starting_pitcher_dob'])
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
    #    inplace=True)
              
    return data


if __name__ == "__main__":


    # ----------  ----------  ----------
    # Parameters
    date = dt.datetime(year=2019, month=6, day=30)
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
    # Get Historic Boxscores
    df_box_hist = get_hist_boxscores(str(date.year))
    df_prev_game_ids = get_prev_game_id(df_box_hist, date)
    
    # ---------- ---------- ----------
    # Get Basic Matchup Table
    df_matchup_base = get_matchup_base_table(date=date)
    df_matchup_base = add_date(df_matchup_base)

    # Narrow to immediate dimensions
    df_matchup_base = df_matchup_base.loc[:, [
        'gameId', 'gameDate',
        'startingPitcherId', 'probableStarterSide'
    ]]

    # # # # 
    # THIS PART IS DIFFERENT FOR TODAY THAN HISTORIC PREP
    # # # #
    # Only keep starting pitcher id that are filled
    df_matchup_base = df_matchup_base.loc[
        df_matchup_base['startingPitcherId'].notnull(), :]

    # Only keep gameIds with both home and away
    vc = df_matchup_base['gameId'].value_counts()
    vc = pd.DataFrame(vc).reset_index(inplace=False)
    vc.columns = ['gameId', 'freq']
    gids = list(vc.loc[vc['freq'] == 2, :]['gameId'])
    df_matchup_base = df_matchup_base.loc[
        df_matchup_base['gameId'].isin(gids), :]

    # Pivot out to flatten
    df_matchup_base = df_matchup_base.pivot_table(
        index=['gameId', 'gameDate'],
        columns=['probableStarterSide'],
        values=['startingPitcherId'],
        aggfunc='first'
    )
    df_matchup_base.reset_index(inplace=True)
    df_matchup_base.columns = [
        x[0] if x[1] == '' else x[0]+"_"+x[1] for
        x in df_matchup_base.columns
    ]

    # Rename
    df_matchup_base.rename(
        columns={
            'startingPitcherId_away': 'away_starting_pitcher',
            'startingPitcherId_home': 'home_starting_pitcher'
        },
        inplace=True
    )

    # # # #
    # BACK TO MATCHING HIST
    # # # #
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
    df_matchup_base.to_csv(
        '/Users/peteraltamura/Desktop/df_matchup_base_tomorrow_details.csv',
        index=False
    )

    

    # ----------  ----------  ----------
    # Add Home starter trailing stats
    
    # Add Away Starter trailing stats
    
