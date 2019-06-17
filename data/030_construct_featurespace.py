import os
import gc
import sys
import numpy as np
import pandas as pd
import datetime as dt
import utilities as util
CONFIG = util.load_config()


def get_prev_batting_game_id(year):
    """
    """
    
    # determine gameId and date combinations we have batting and pitching for
    batting_gameId_dates = pd.concat(
            objs=[
                pd.read_parquet(CONFIG.get('paths').get('batter_saber')+
                                d+"/batter_saber.parquet",
                                columns=['gameId', 'gameDate'])
                for d in os.listdir(CONFIG.get('paths').get('batter_saber'))
                if os.path.isfile(CONFIG.get('paths').get('batter_saber')+d+
                                  '/batter_saber.parquet')
                and year in d
            ],
            axis=0
    )
    batting_gameId_dates.loc[:, 'gameDate'] = pd.to_datetime(
            batting_gameId_dates['gameDate'],
            infer_datetime_format=True
    )
    batting_gameId_dates.drop_duplicates(inplace=True)
    return batting_gameId_dates


def get_prev_pitching_game_id(year):
    """
    """

    # determine gameid and date combinations we have batting and pitching for
    pitching_gameId_dates = pd.concat(
            objs=[
                pd.read_parquet(CONFIG.get('paths').get('pitcher_saber')+
                                d+"/pitcher_saber.parquet",
                                columns=['gameId', 'gameDate'])
                for d in os.listdir(CONFIG.get('paths').get('pitcher_saber'))
                if os.path.isfile(CONFIG.get('paths').get('pitcher_saber')+d+
                                  '/pitcher_saber.parquet')
                and year in d
            ],
            axis=0
    )
    pitching_gameId_dates.loc[:, 'gameDate'] = pd.to_datetime(
            pitching_gameId_dates['gameDate'],
            infer_datetime_format=True
    )
    pitching_gameId_dates.drop_duplicates(inplace=True)
    return pitching_gameId_dates


def get_prev_game_id(data):
    """
    """

    # Team Tables
    team_tables = []

    # Sort gameIds for each team
    for team in list(set(
            list(data['away_code']) +
            list(data['home_code'])
    )):
        # Combine Home and Away
        team_data = data.loc[data['gameId'].str.contains(team), :]
        team_data.sort_values(by=['gameId'], ascending=True, inplace=True)
        team_data.loc[:, 'prevGameId'] = team_data['gameId'].shift(1)
        team_data['team_code'] = team
        team_tables.append(team_data[['gameId', 'team_code', 'prevGameId']])   
    team_tables = pd.concat(
        objs=team_tables,
        axis=0
    )
    return team_tables


def add_starting_pitcher_id(data, year):
    """
    """
    # determine gameid and date combinations we have batting and pitching for
    inning_tables = pd.concat(
        objs=[
            pd.read_parquet(
                CONFIG.get('paths').get('normalized')+
                d+"/innings.parquet",
                columns=['gameId', 'home_starting_pitcher',
                             'away_starting_pitcher']
            )
            for d in os.listdir(CONFIG.get('paths').get('normalized'))
            if year in d and 
            os.path.isfile(
                CONFIG.get('paths').get('normalized')+d+
                '/innings.parquet'
            )
        ],
        axis=0
    )
    inning_tables.drop_duplicates(inplace=True)
    data = pd.merge(
        data,
        inning_tables,
        how='left',
        on=['gameId'],
        validate='1:1'
    )
    return inning_tables


def win_loss_by_disposition(data):
    """
    """

    # Set dictionary to add to
    dic = {}

    # List teams
    teams = list(set(
        list(set(data['away_code'])) +
        list(set(data['home_code']))
    ))

    # Add Date
    data['gameDate'] = data['gameId'].str[4:14]
    data['gameDate'] = pd.to_datetime(data['gameDate'], format="%Y_%m_%d")

    # Add Flags (duplicative)
    data['home_team_winner'] = (data['home_team_runs'] >
                                data['away_team_runs']).astype(int)
    data['away_team_winner'] = (data['away_team_runs'] >
                                data['home_team_runs']).astype(int)

    # Iteration
    home_team_stats_tables = []
    away_team_stats_tables = []
    for team in teams:

        print(team)
        print(data.shape)
        base_left = data.loc[:, ['gameId']]

        # Home
        curr = data.loc[data['gameId'].str.contains(team), :]
        curr = curr.loc[curr['home_code'] == team, :]
        curr['disposition'] = 'home'
        curr.sort_values(by=['gameId'], ascending=True, inplace=True)
        curr.loc[:, 'home_team_winners'] = curr['home_team_winner'].cumsum()
        curr.reset_index(drop=True, inplace=True)
        curr.loc[:, 'home_team_win_pct_at_home'] = (
            curr['home_team_winners'] / (
                curr.index + 1
            )
        )
        if curr.shape[0]== 0:
            continue
        
        curr = curr.loc[:, [
            'gameId', 'disposition', 'home_team_win_pct_at_home'
        ]]
        curr.loc[:, 'away_team_win_pct_at_away'] = np.NaN
        home_team_stats_tables.append(curr)

        # Away
        curr = data.loc[data['gameId'].str.contains(team), :]
        curr = curr.loc[curr['away_code'] == team, :]
        curr['disposition'] = 'away'
        curr.sort_values(by=['gameId'], ascending=True, inplace=True)
        curr.loc[:, 'away_team_winners'] = curr['away_team_winner'].cumsum()
        curr.reset_index(drop=True, inplace=True)
        curr.loc[:, 'away_team_win_pct_at_away'] = (
            curr['away_team_winners'] / (
                curr.index + 1
            )
        )
        curr = curr.loc[:, ['gameId', 'disposition', 'away_team_win_pct_at_away']]
        if curr.shape[0] == 0:
            continue
        curr.loc[:, 'home_team_win_pct_at_home'] = np.NaN
        away_team_stats_tables.append(curr)

    # Combine
    home_team_stats_tables = pd.concat(
        objs=home_team_stats_tables,
        axis=0
    )
    away_team_stats_tables = pd.concat(
        objs=away_team_stats_tables,
        axis=0
    )
    home_team_stats_tables = home_team_stats_tables.loc[:, [
        'gameId', 'disposition', 'home_team_win_pct_at_home'
    ]]
    away_team_stats_tables = away_team_stats_tables.loc[:, [
        'gameId', 'disposition', 'away_team_win_pct_at_away'
    ]]
    return home_team_stats_tables, away_team_stats_tables


def get_full_boxscores(team):
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
              )]
    for fname in fnames:
        df = pd.read_parquet(fname)
        df = df.loc[df['gameId'].str.contains(team), :]
        df = df.loc[:, [
            'gameId', 'date', 'away_team_flag', 'home_team_flag',
            'away_wins', 'away_loss', 'home_wins', 'home_loss',
        ]]
        df_boxscores.append(df)
    df_boxscores = pd.concat(
        objs=df_boxscores,
        axis=0
    )
    return df_boxscores


def get_full_linescore_summaries(team):
    """
    """
    df_linescore_summaries = []
    fnames = [
        CONFIG.get('paths').get('normalized')+dstr+"/game_linescore_summary.parquet"
        for dstr in os.listdir(CONFIG.get('paths').get('normalized')) if
        os.path.isfile(CONFIG.get('paths').get('normalized')+dstr+
                       "/game_linescore_summary.parquet")
    ]
    for fname in fnames:
        df = pd.read_parquet(fname)
        df = df.loc[df['gameId'].str.contains(team), :]
        df_linescore_summaries.append(df)
    df_linescore_summaries = pd.concat(
        objs=df_linescore_summaries,
        axis=0
    )
    return df_linescore_summaries


def add_batting_sabermetrics(team, batter_metrics):
    """
    """
    path = CONFIG.get('paths').get('batter_saber')
    batting = [
        path+dd+"/batter_saber.parquet" for dd in
        os.listdir(path) if team in dd
    ]
    df = pd.concat(
        objs=[pd.read_parquet(bat_path) for bat_path
              in batting],
        axis=0
    )
    df = pd.pivot_table(
        index=['gameId'],
        columns=['ab_trail3'],
        values=[
            x for x in df.columns if any(
            metric in x for metric in batter_metrics)
        ],
        aggfunc='first'
    )
    return df


def add_pitching_sabermetrics(team, pitcher_metrics):
    """
    """
    path = CONFIG.get('paths').get('pitcher_saber')
    pitching = [
        path+dd+"/pitcher_saber.parquet" for dd in
        os.listdir(path) if team in dd
    ]
    df = pd.concat(
        objs=[pd.read_parquet(pitch_path) for pitch_path
              in pitching],
        axis=0
    )

    # Add the rest
    df = df.pivot_table(
        index=['gameId'],
        columns=['bf_trail3'],
        values=[
            x for x in df.columns if any(
                metric in x for metric in pitcher_metrics
            )
        ],
        aggfunc='first'
    )
    return df


def add_starting_pitcher_sabermetrics(data, team, pitcher_metrics):
    """
    """

    path = CONFIG.get('paths').get('pitcher_saber')
    pitching = [
        path+dd+"/pitcher_saber.parquet" for dd in
        os.listdir(path) if team in dd
    ]
    df = pd.concat(
        objs=[pd.read_parquet(pitch_path) for pitch_path
              in pitching],
        axis=0
    )

    # Determine whether to add home starter or away starter
    # if in 5, home
    if team in df['gameId'].iloc[0].split("_")[5]:
        starter_disposition = 'home_starting_pitcher'
    elif team in df['gameId'].iloc[0].split("_")[4]:
        starter_disposition = 'away_starting_pitcher'
    else:
        raise Exception("Home or Away Disposition Still Unknown")

    # Cut down to gameId, pitcherId, and stats
    df = df.loc[:, ['gameId', 'pitcherId'] + [
        x for x in df.columns if any(
            metric in x for metric in pitcher_metrics
        )
    ]]
    data = pd.merge(
        data,
        df,
        how='left',
        left_on=['gameId', starter_disposition],
        right_on=['gameId', 'pitcherId'],
        validate='1:1'
    )
    return data
    



if __name__ == "__main__":

    # Full dataset path
    matchups_path = CONFIG.get('paths').get('matchup_schedule')
    years = list(set([x[:4] for x in os.listdir(matchups_path)]))
    years = [
        str(yr) for yr in years if str(yr) in [
            str(y) for y in np.arange(1967, 2020, 1)
        ]
    ]
    top_pitcher_count = 8
    pitcher_metrics = ['BF', 'ER', 'ERA', 'HitsAllowed', 'Holds',
                       'SeasonLosses', 'SeasonWins', 'numberPitches',
                       'Outs', 'RunsAllowed', 'Strikes', 'SO']
    top_batter_count = 12
    batter_metrics = [
        'batterWalkPercentage',	'batterKPercentage', 'batterISO',
        'batterBABIP', 'woba'
    ]


    # ------------------------------
    # Read in base table from schedule
    for yr in ['2017']:
        
        # Read in current year Summaries (will be cut down later)
        df_base = pd.concat(
            objs=[
                pd.read_parquet(
                    CONFIG.get('paths').get('normalized') + \
                    dd + "/" + "game_linescore_summary.parquet"
                ) for dd in os.listdir(
                    CONFIG.get('paths').get('normalized')
                ) if str(yr) in dd and
                os.path.isfile(CONFIG.get('paths').get('normalized') + \
                    dd + "/" + "game_linescore_summary.parquet")
            ],
            axis=0
        )

        # Prev Batting and Pitching GameID and Filter
        prev_batting_ids = get_prev_batting_game_id(yr)
        prev_pitching_ids = get_prev_pitching_game_id(yr)
        df_base = df_base.loc[(
            (df_base['gameId'].isin(list(prev_batting_ids['gameId'])))
            &
            (df_base['gameId'].isin(list(prev_pitching_ids['gameId'])))
        ), :]
        del prev_batting_ids
        del prev_pitching_ids
        gc.collect()

        # Get Previous games of those remaining
        prev_game_ids = get_prev_game_id(df_base)

        # Win Loss by Disposition
        home_team_win_pct_home, away_team_win_pct_away = win_loss_by_disposition(df_base)

        # ---------------------------------------------
        # Iterate over teams and apend final table to list
        complete_team_tables = []
        teams_list = list(set(
            list(df_base['away_code'])+
            list(df_base['home_code'])
        ))
        
        for team in teams_list:
            if team in ['aas', 'nas', 'umi', 'atf', 'lvg']:
                continue
            
            print("Now creating featurespace for team: {}".format(team))

            # Subset base for current team
            df_base_curr = df_base.loc[df_base['gameId'].str.contains(team), :]

            # Add starting pitcher from Normalized Innings table
            df_base_curr = add_starting_pitcher_id(df_base_curr, yr)

            # Add previous gameIds to home
            df_base_curr = pd.merge(
                df_base_curr,
                prev_game_ids,
                how='left',
                left_on=['gameId', 'home_code'],
                right_on=['gameId', 'team_code'],
                validate='1:1'
            )
            df_base_curr.rename(
                columns={'prevGameId': 'homePrevGameId'},
                inplace=True
            )

            # Add prev gameIds to away
            df_base_curr = pd.merge(
                df_base_curr,
                prev_game_ids,
                how='left',
                left_on=['gameId', 'away_code'],
                right_on=['gameId', 'team_code'],
                validate='1:1'
            )
            df_base_curr.rename(
                columns={'prevGameId': 'awayPrevGameId'},
                inplace=True
            )

            # Merge home team win pct at home table
            df_base_curr.drop_duplicates(subset=['homePrevGameId'], inplace=True)
            df_base_curr = pd.merge(
                df_base_curr,
                home_team_win_pct_home,
                how='left',
                left_on=['homePrevGameId'],
                right_on=['gameId'],
                validate='1:1',
                suffixes=['', '_del']
            )
            df_base_curr.drop(
                labels=[x for x in df_base_curr.columns if x[-4:] == '_del'],
                axis=1,
                inplace=True
            )

            # Merge away team win pct at away table
            df_base_curr.drop_duplicates(subset=['awayPrevGameId'], inplace=True)
            df_base_curr = pd.merge(
                df_base_curr,
                away_team_win_pct_away,
                how='left',
                left_on=['awayPrevGameId'],
                right_on=['gameId'],
                validate='1:1',
                suffixes=['', '_del']
            )
            df_base_curr.drop(
                labels=[x for x in df_base_curr.columns if x[-4:] == '_del'],
                axis=1,
                inplace=True
            )
            
            # Merge on Winner
            df_linescore_summary = get_full_linescore_summaries(team)
            df_linescore_summary = df_linescore_summary.loc[:, [
                'gameId', 'away_team_runs', 'home_team_runs'
            ]]
            df_linescore_summary.loc[:, 'home_team_win'] = (
                df_linescore_summary['home_team_runs'] >
                df_linescore_summary['away_team_runs']
            ).astype(int)
            df_base_curr = pd.merge(
                df_base_curr,
                df_linescore_summary,
                how='left',
                left_on=['gameId'],
                right_on=['gameId'],
                validate='1:1'
            )

            # Add Starter Sabermetrics from pitchers_wide
            df_base_curr = add_starting_pitcher_sabermetrics(
                df_base_curr, team, pitcher_metrics
            )

            # Add batting Sabermetrics
            batter_wide = add_batting_sabermetrics(team, batter_metrics)

            # Add pitching Sabermetrics
            pitcher_wide = add_pitching_sabermetrics(team, pitcher_metrics)
            
