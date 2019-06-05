import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
import utilities as util
CONFIG = util.load_config()


def get_batting_games_dates():
    # determine gameId and date combinations we have batting and pitching for
    batting_gameId_dates = pd.concat(
            objs=[
                pd.read_parquet(CONFIG.get('paths').get('batter_stats')+
                                d+"/batter_stats.parquet",
                                columns=['gameId', 'gameDate'])
                for d in os.listdir(CONFIG.get('paths').get('batter_stats'))
                if os.path.isfile(CONFIG.get('paths').get('batter_stats')+d+'/batter_stats.parquet')
            ],
            axis=0
    )
    batting_gameId_dates.loc[:, 'gameDate'] = pd.to_datetime(
            batting_gameId_dates['gameDate'],
            infer_datetime_format=True
    )
    batting_gameId_dates.drop_duplicates(inplace=True)
    return batting_gameId_dates


def get_pitching_games_dates():
    """
    """
    pitching_gameId_dates = pd.concat(
            objs=[
                pd.read_parquet(CONFIG.get('paths').get('pitcher_stats')+
                                d+"/pitcher_stats.parquet",
                                columns=['gameId', 'gameDate'])
                for d in os.listdir(CONFIG.get('paths').get('pitcher_stats'))
                if os.path.isfile(CONFIG.get('paths').get('pitcher_stats')+d+'/pitcher_stats.parquet')
            ],
            axis=0
    )
    pitching_gameId_dates.loc[:, 'gameDate'] = pd.to_datetime(
            pitching_gameId_dates['gameDate'],
            infer_datetime_format=True
    )
    pitching_gameId_dates.drop_duplicates(inplace=True)
    return pitching_gameId_dates


def get_full_batting_stats(team):
    """
    """

    df_batting = []
    fnames = [CONFIG.get('paths').get('batter_stats')+gid+"/batter_stats.parquet"
              for gid in os.listdir(CONFIG.get('paths').get('batter_stats'))]
    for fname in fnames:
        df = pd.read_parquet(fname)
        df = df.loc[df['gameId'].str.contains(team), :]
        df_batting.append(df)
    df_batting = pd.concat(
        objs=df_batting,
        axis=0
    )
    return df_batting


def get_full_pitching_stats(team):
    """
    """

    df_pitching = []
    fnames = [CONFIG.get('paths').get('pitcher_stats')+gid+"/pitcher_stats.parquet"
              for gid in os.listdir(CONFIG.get('paths').get('pitcher_stats'))]
    for fname in fnames:
        df = pd.read_parquet(fname)
        df = df.loc[df['gameId'].str.contains(team), :]
        df_pitching.append(df)
    df_pitching = pd.concat(
        objs=df_pitching,
        axis=0
    )
    return df_pitching


def get_full_boxscores(team):
    """
    """
    df_boxscores = []
    fnames = [CONFIG.get('paths').get('normalized')+dstr+"/boxscore.parquet"
              for dstr in os.listdir(CONFIG.get('paths').get('normalized'))
              if os.path.isfile(CONFIG.get('paths').get('normalized')+dstr+"/boxscore.parquet")]
    for fname in fnames:
        df = pd.read_parquet(fname)
        df = df.loc[df['gameId'].str.contains(team), :]
        df = df.loc[:, [
            'gameId', 'away_team_flag', 'home_team_flag',
            'away_wins', 'away_loss', 'home_wins', 'home_loss',
            
            
        df_boxscores.append(df)
    df_boxscores = pd.concat(
        objs=df_boxscores,
        axis=0
    )
    return df_boxscores


if __name__ == "__main__":

    # Full dataset path
    matchups_path = CONFIG.get('paths').get('matchup_schedule')
    prepared_fs_path = CONFIG.get('paths').get('prepared_featurespaces')
    years = list(set([x[:4] for x in os.listdir(matchups_path)]))
    years = [str(yr) for yr in years if str(yr) in [str(y) for y in np.arange(1967, 2020, 1)]]

    # ----------
    # Read in base table from schedule
    for yr in ['2018']:

        # Read in current year matchup schedule
        df_base = pd.read_csv(matchups_path+'{}_matchups.csv'.format(yr), dtype=str)
        df_base.loc[:, 'date'] = pd.to_datetime(df_base['date'], infer_datetime_format=True)
        
        # Read in prepared featurespaces from current year
        if "{}_featurespace.parquet" in os.listdir(prepared_fs_path):
            df_features = pd.read_parquet(
                prepared_fs_path+"{}_featurespace.parquet".format(yr)
            )
            df_base = pd.merge(
                df_base,
                df_features,
                how='left',
                on=['gameId'],
                validate='1:1'
            )
            df_base = df_base.loc[df_base['batterId'].notnull(), :]

        # ------------------------------
        # Continue 
        batting_gameId_dates = get_batting_games_dates()
        pitching_gameId_dates = get_pitching_games_dates()

        # Inner merge to base from batting
        df_base = pd.merge(
            df_base,
            batting_gameId_dates,
            how='inner',
            left_on=['gameId', 'date'],
            right_on=['gameId', 'gameDate'],
            validate='1:1'
        )
        df_base.drop(labels=['gameDate'], axis=1, inplace=True)

        # Innter merge to base from pitching
        df_base = pd.merge(
            df_base,
            pitching_gameId_dates,
            how='inner',
            left_on=['gameId', 'date'],
            right_on=['gameId', 'gameDate'],
            validate='1:1'
        )
        df_base.drop(labels=['gameDate'], axis=1, inplace=True)
        
        # Get list of teams
        complete_team_tables = []
        teams_list = list(set(
            list(df_base['away_team_code'])+
            list(df_base['home_team_code'])
        ))

        # Iterate over teams and apend final table to list
        for team in teams_list:
            print(team)
            
            # Rank and order the gameIds we have stats for
            df_base_curr = df_base.loc[df_base['gameId'].str.contains(team), :]
            df_base_curr.sort_values(by=['date'], ascending=True, inplace=True)
            df_base_curr.loc[:, 'obsv_rank'] = np.arange(df_base_curr.shape[0])
            df_base_curr.loc[:, 'obsv_rank'] += 1
            assert min(df_base_curr['obsv_rank']) == 1

            # Filter to team games for batting, sort and merge
            team_batting = get_full_batting_stats(team)
            team_batting = team_batting.loc[
                team_batting['gameId'].isin(list(set(df_base_curr['gameId']))),
            :]
            team_batting.sort_values(by=['gameDate'], ascending=True, inplace=True)
            team_batting['obsv_rank'] = np.arange(team_batting.shape[0])
            team_batting.drop(labels=['gameId', 'gameDate'], axis=1, inplace=True)
            df_base_curr = pd.merge(df_base_curr, team_batting, how='left', on=['obsv_rank'], validate='1:1')
            
            # Filter to team games for pitching
            team_pitching = get_full_pitching_stats(team)
            team_pitching = team_pitching.loc[
                team_pitching['gameId'].isin(list(set(df_base_curr['gameId']))),
            :]
            team_pitching.sort_values(by=['gameDate'], ascending=True, inplace=True)
            team_pitching['obsv_rank'] = np.arange(team_pitching.shape[0])
            team_pitching.drop(labels=['gameId', 'gameDate'], axis=1, inplace=True)
            df_base_curr = pd.merge(df_base_curr, team_pitching, how='left', on=['obsv_rank'], validate='1:1')

            # Add score from game
            boxscores = get_full_boxscores(team)

            # Create flag
            

            df_base_curr.to_parquet(
                CONFIG.get('paths').get('initial_featurespaces') + \
                '{}_{}_initial_featurespace.parquet'.format(str(yr), str(team))
            )

    
