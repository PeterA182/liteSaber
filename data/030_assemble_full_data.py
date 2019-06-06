import os
import gc
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
    fnames = [f for f in fnames if team in f]
    for fname in fnames:
        df = pd.read_parquet(fname)
        print(df.shape)
        print(list(set(df['gameId'])))
        print(team)
        df = df.loc[df['gameId'].str.contains(team), :]
        print(df.shape)
        print(list(pd.Series.unique(df['gameId'])))
        assert pd.Series.nunique(df['gameId']) == 1
        gameId = df['gameId'].iloc[0]
        if team in gameId.split("_")[4]:
            df = df.loc[df['batterTeamFlag'] == 'away', :]
        elif team in gameId.split("_")[5]:
            df = df.loc[df['batterTeamFlag'] == 'home', :]
        else:
            raise
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
    fnames = [f for f in fnames if team in f]
    for fname in fnames:
        df = pd.read_parquet(fname)
        df = df.loc[df['gameId'].str.contains(team), :]
        gameId = df.iloc['gameId']
        if gameId.split("_")[4] == team:
            df = df.loc[df['pitcherTeamFlag'] == 'away', :]
        elif gameId.split("_")[5] == team:
            df = df.loc[df['pitcherTeamFlag'] == 'home', :]
        else:
            raise
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
              if os.path.isfile(CONFIG.get('paths').get('normalized')+
                                dstr+"/boxscore.parquet")]
    for fname in fnames:
        df = pd.read_parquet(fname)
        df = df.loc[df['gameId'].str.contains(team), :]
        df = df.loc[:, [
            'gameId', 'away_team_flag', 'home_team_flag',
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
        CONFIG.get('paths').get('raw')+dstr+"/game_linescore_summary.parquet"
        for dstr in os.listdir(CONFIG.get('paths').get('raw')) if
        os.path.isfile(CONFIG.get('paths').get('raw')+dstr+"/game_linescore_summary.parquet")
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


def pivot_stats_wide(data, swing_col, metric_cols):
    """
    """

    # Assemble proper metrics list
    if swing_col == 'batterId':
        metric_cols = [
            x for x in data.columns if any(
                y in x for y in [
                    'batter{}'.format(xx)
                    for xx in metric_cols
                ]
            )
        ]
    elif swing_col == 'pitcherId':
        metric_cols = [
            x for x in data.columns if any(
                y in x for y in [
                    'pitcher{}'.format(xx)
                    for xx in metric_cols
                ]
            )
        ]
    else:
        raise

    # Pivot Table
    player_ids = list(set(data[swing_col]))
    data = data.pivot_table(
        index=['gameId'],
        columns=[swing_col],
        values=metric_cols
    )
    data.reset_index(inplace=True)
    data.columns = [
        x[0] if x[1] == '' else x[0]+"_"+str(player_ids.index(x[1]))
        for x in data.columns
    ]
    print(data.columns)
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
    batter_metrics = ['H', ]
    pitcher_metrics = ['BF', 'ER', 'ERA', 'HitsAllowed', 'Holds',
                       'SeasonLosses', 'SeasonWins', 'numberPitches',
                       'Outs', 'RunsAllowed', 'Strikes', 'SO']
    batter_metrics = ['Assists', 'AB', 'BB', 'FO', 'Avg', 'H', 'HBP', 'HR',
                      'Doubles' 'GroundOuts', 'batterLob',
                      'OBP', 'OPS', 'R', 'RBI', 'SluggingPct',
                      'StrikeOuts', 'Triples']    

    # ----------
    # Read in base table from schedule
    for yr in ['2018']:

        # Read in current year matchup schedule
        df_base = pd.read_csv(
            matchups_path+'{}_matchups.csv'.format(yr),
            dtype=str
        )
        df_base.loc[:, 'date'] = pd.to_datetime(
            df_base['date'],
            infer_datetime_format=True
        )

        #
        #
        # ---------------------------------------------
        # ---------------------------------------------
        # PREPARATION TO ENSURE ALL MERGES ARE COMPLETE

        # ---------------------------------------------
        # Inner merge to base from batting
        # to make sure all gameIds are in both
        batting_gameId_dates = get_batting_games_dates()
        df_base = pd.merge(
            df_base,
            batting_gameId_dates,
            how='inner',
            left_on=['gameId', 'date'],
            right_on=['gameId', 'gameDate'],
            validate='1:1'
        )
        del batting_gameId_dates
        gc.collect()
        df_base.drop(labels=['gameDate'], axis=1, inplace=True)

        # ---------------------------------------------
        # Inner merge to base from pitching
        # to make sure all gameIds are in both
        pitching_gameId_dates = get_pitching_games_dates()
        df_base = pd.merge(
            df_base,
            pitching_gameId_dates,
            how='inner',
            left_on=['gameId', 'date'],
            right_on=['gameId', 'gameDate'],
            validate='1:1'
        )
        del pitching_gameId_dates
        gc.collect()
        df_base.drop(labels=['gameDate'], axis=1, inplace=True)

        
        # Get list of teams to iterate over
        teams_list = list(set(
            list(df_base['away_team_code'])+
            list(df_base['home_team_code'])
        ))

        #
        #
        # ---------------------------------------------
        # ---------------------------------------------
        # PREPARATION TO ENSURE ALL MERGES ARE COMPLETE
        # Iterate over teams and apend final table to list
        complete_team_tables = []
        for team in teams_list:
            print("Now creating featurespace for team: {}".format(team))

            # --------------------
            # Get sorted list of gameIds for team
            team_gameids = df_base.loc[df_base['gameId'].str.contains(team), :]
            team_gameids.sort_values(
                by=['date'],
                ascending=True,
                inplace=True
            )
            team_gameids = list(team_gameids['gameId'])

            # --------------------
            # Subset and sort base table for current team
            df_base_curr = df_base.loc[
                df_base['gameId'].str.contains(team), :]
            df_base_curr.sort_values(
                by=['date'], ascending=True, inplace=True
            )
            df_base_curr.loc[:, 'prev_gameid_merge_key'] = \
                df_base_curr['gameId'].apply(
                    lambda x: (
                        team_gameids[team_gameids.index(x)-1] if
                        team_gameids.index(x) > 0 else 'nan'
                    )
                )

            # --------------------
            # Filter to team games for batting, sort and merge
            team_batting = get_full_batting_stats(team)
            team_batting = team_batting.loc[
                team_batting['gameId'].isin(
                    list(set(df_base_curr['gameId']))
                ),
            :]
            team_batting.sort_values(
                by=['gameDate'], ascending=True, inplace=True
            )
            team_batting.drop(labels=['gameDate'],
                              axis=1, inplace=True)
            team_batting = pivot_stats_wide(team_batting,
                                            swing_col='batterId',
                                            metric_cols=batter_metrics)
            df_base_curr = pd.merge(
                df_base_curr, team_batting,
                how='left',
                left_on=['prev_gameid_merge_key'],
                right_on=['gameId'],
                validate='1:1'
            )
            #df_base_curr.drop(labels=['gameId'], axis=1, inplace=True)
            
            # Filter to team games for pitching
            team_pitching = get_full_pitching_stats(team)
            team_pitching = team_pitching.loc[
                team_pitching['gameId'].isin(list(set(df_base_curr['gameId']))),
            :]
            team_pitching.sort_values(
                by=['gameDate'],
                ascending=True,
                inplace=True
            )
            team_pitching.drop(
                labels=['gameDate'],
                axis=1,
                inplace=True
            )
            team_pitching = pivot_stats_wide(team_batting,
                                             swing_col='pitcherId',
                                             metric_cols=pitcher_metrics)
            df_base_curr = pd.merge(
                df_base_curr, team_pitching,
                how='left',
                left_on=['prev_gameid_merge_key'],
                right_on=['gameId'],
                validate='1:1'
            )
            #df_base_curr.drop(labels=['gameId'], axis=1, inplace=True)

            # Add scores from boxscores
            boxscores = get_full_boxscores(team)
            boxscores = boxscores.loc[
                boxscore['gameId'].isin(
                    list(set(df_base_curr['gameId']))
                ),
            :]
            boxscores.sort_values(
                by=['gameDate'], ascending=True, inplace=True
            )
            df_base_curr = pd.merge(
                df_base_curr, boxscores,
                how='left',
                left_on=['prev_gameid_merge_key'],
                right_on=['gameId'],
                validate='1:1'
            )

            # Add

            # Create flag
            df_base_curr.to_parquet(
                CONFIG.get('paths').get('initial_featurespaces') + \
                '{}_{}_initial_featurespace.parquet'.format(
                    str(yr), str(team)
                )
            )

    
