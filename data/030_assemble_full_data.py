import os
import gc
import sys
import numpy as np
import pandas as pd
import datetime as dt
import utilities as util
CONFIG = util.load_config()


def get_merge_key_dict(data):
    """
    """

    # Set dictionary to add to
    dic = {}
    
    # Sort gameIds for each team
    for team in list(set(
            list(data['away_code']) +
            list(data['home_code'])
    )):
        
        team_data = data.loc[data['gameId'].str.contains(team), :]
        team_data.sort_values(by=['gameId'], ascending=False, inplace=True)
        team_data.loc[:, 'prevGameId'] = team_data['gameId'].shift(1)
        team_data.loc[:, 'team_abbrev'] = team
        print(team)
        print(np.mean(team_data['gameId'].isnull()))
        print(np.mean(team_data['team_abbrev'].isnull()))
        team_data.loc[:, 'zip'] = team_data[['gameId', 'team_abbrev']].apply(tuple, axis=1)
        team_data = team_data.set_index('zip')['prevGameId'].to_dict()
        dic.update(team_data)

    return dic


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
    tables = []
    for team in teams:
        curr = data.loc[data['gameId'].str.contains(team), :]
        curr.sort_values(by=['gameDate'], ascending=True, inplace=True)
        curr['home_win_cumsum'] = curr['home_team_winner'].cumsum().shift(1)
        curr['away_win_cumsum'] = curr['away_team_winner'].cumsum().shift(1)
        curr = curr.loc[:, ['gameId', 'home_win_cumsum', 'away_win_cumsum']]
        tables.append(curr)
    tables = pd.concat(
        objs=tables,
        axis=0
    )
    return tables
    


def get_batting_games_dates():
    # determine gameId and date combinations we have batting and pitching for
    batting_gameId_dates = pd.concat(
            objs=[
                pd.read_parquet(CONFIG.get('paths').get('batter_stats')+
                                d+"/batter_stats.parquet",
                                columns=['gameId', 'gameDate'])
                for d in os.listdir(CONFIG.get('paths').get('batter_stats'))
                if os.path.isfile(CONFIG.get('paths').get('batter_stats')+d+
                                  '/batter_stats.parquet')
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
                if os.path.isfile(CONFIG.get('paths').get('pitcher_stats')+d+
                                  '/pitcher_stats.parquet')
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
        df = df.loc[df['gameId'].str.contains(team), :]
        if pd.Series.nunique(df['gameId']) != 1:
            return pd.DataFrame()
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


def batter_appearance_freq(data, top_batter_count):
    """
    """

    # Rank batterId by most appearances and grab those in count
    freq = data.groupby(
        by=['batterId'],
        as_index=False
    ).agg({'gameId': pd.Series.nunique})
    freq.sort_values(by=['gameId'], ascending=False, inplace=True)
    freq = freq.head(top_batter_count)
    freq_batters = list(freq['batterId'])
    data = data.loc[data['batterId'].isin(freq_batters), :]
    return data


def get_full_pitching_stats(team):
    """
    """

    df_pitching = []
    fnames = [CONFIG.get('paths').get('pitcher_stats')+gid+
              "/pitcher_stats.parquet"
              for gid in os.listdir(
                      CONFIG.get('paths').get('pitcher_stats')
              )]
    fnames = [f for f in fnames if team in f]
    for fname in fnames:
        df = pd.read_parquet(fname)
        df = df.loc[df['gameId'].str.contains(team), :]
        if pd.Series.nunique(df['gameId']) != 1:
            return pd.DataFrame()
        gameId = df['gameId'].iloc[0]
        if team in gameId.split("_")[4]:
            df = df.loc[df['pitcherTeamFlag'] == 'away', :]
        elif team in gameId.split("_")[5]:
            df = df.loc[df['pitcherTeamFlag'] == 'home', :]
        else:
            raise
        df_pitching.append(df)
    df_pitching = pd.concat(
        objs=df_pitching,
        axis=0
    )
    return df_pitching


def pitcher_appearance_freq(data, top_pitcher_count):
    """
    """

    # Rank batterId by most appearances and grab those in count
    freq = data.groupby(
        by=['pitcherId'],
        as_index=False
    ).agg({'gameId': pd.Series.nunique})
    freq.sort_values(by=['gameId'], ascending=False, inplace=True)
    freq = freq.head(top_pitcher_count)
    freq_pitchers = list(freq['pitcherId'])
    data = data.loc[data['pitcherId'].isin(freq_pitchers), :]
    return data

    

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
        CONFIG.get('paths').get('raw')+dstr+"/game_linescore_summary.parquet"
        for dstr in os.listdir(CONFIG.get('paths').get('raw')) if
        os.path.isfile(CONFIG.get('paths').get('raw')+dstr+
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
    print(sorted(data.columns))
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
    top_pitcher_count = 15
    pitcher_metrics = ['BF', 'ER', 'ERA', 'HitsAllowed', 'Holds',
                       'SeasonLosses', 'SeasonWins', 'numberPitches',
                       'Outs', 'RunsAllowed', 'Strikes', 'SO']
    top_batter_count = 15
    batter_metrics = ['Assists', 'AB', 'BB', 'FO', 'Avg', 'H',
                      'HBP', 'HR', 'Doubles' 'GroundOuts', 'batterLob',
                      'OBP', 'OPS', 'R', 'RBI', 'SluggingPct',
                      'StrikeOuts', 'Triples']    
    
    # ----------
    # Read in base table from schedule
    for yr in ['2018']:

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

        #
        # ---------------------------------------------
        # PREPARATION TO ENSURE ALL MERGES ARE COMPLETE

        # ---------------------------------------------
        # Inner merge to base from batting
        # to make sure all gameIds are in both
        batting_gameId_dates = get_batting_games_dates()
        batting_dates = list(set(batting_gameId_dates['gameId']))
        df_base = df_base.loc[df_base['gameId'].isin(batting_dates), :]
        del batting_gameId_dates
        gc.collect()

        # ---------------------------------------------
        # Inner merge to base from pitching
        # to make sure all gameIds are in both
        pitching_gameId_dates = get_pitching_games_dates()
        pitching_dates = list(set(pitching_gameId_dates['gameId']))
        df_base = df_base.loc[df_base['gameId'].isin(pitching_dates), :]
        del pitching_gameId_dates
        gc.collect()
        
        # Get list of teams to iterate over
        teams_list = list(set(
            list(df_base['away_code'])+
            list(df_base['home_code'])
        ))

        # Merge Key Dictionary by Team
        team_prev_merge_key_dict = get_merge_key_dict(df_base)

        # Win Loss by Disposition
        team_win_loss_disp = win_loss_by_disposition(df_base)
        print(team_win_loss_disp.head(24))
        #
        # ---------------------------------------------
        # PREPARATION TO ENSURE ALL MERGES ARE COMPLETE

        # ---------------------------------------------
        # Iterate over teams and apend final table to list
        complete_team_tables = []
        
        #for team in teams_list:
        for team in teams_list:
            print("Now creating featurespace for team: {}".format(team))

            # Subset base for current team
            df_base_curr = df_base.loc[df_base['gameId'].str.contains(team), :]
            print(df_base_curr.shape)

            # Create previous game merge key for home team
            df_base_curr.loc[:, 'zipKey'] = df_base_curr[['gameId', 'home_code']].apply(tuple, axis=1)
            df_base_curr.loc[:, 'awayPrevGameId'] = \
                df_base_curr['zipKey'].map(team_prev_merge_key_dict)
            df_base_curr.drop(labels=['zipKey'], axis=1, inplace=True)
            
            # Create previous game merge key for away team
            df_base_curr.loc[:, 'zipKey'] = df_base_curr[['gameId', 'away_code']].apply(tuple, axis=1)
            df_base_curr.loc[:, 'homePrevGameId'] = \
                df_base_curr['zipKey'].map(team_prev_merge_key_dict)
            df_base_curr.drop(labels=['zipKey'], axis=1, inplace=True)

            #HANDLE THS MERGE
            df_base_curr = pd.merge(
                df_base_curr,
                team_win_loss_disp,
                how='left',
                on=['gameId'],
                validate='1:1'
            )
            print(df_base_curr.shape)
            df_base_curr[['gameId', 'awayPrevGameId', 'homePrevGameId',
                          'away_code', 'home_code', 'away_division',
                          'home_division', 'away_loss', 'home_loss',
                          'away_win', 'away_loss']].to_csv(
                              '/Users/peteraltamura/Desktop/df_base_curr.csv')
            sdkj

            # --------------------
            # Filter to team games for batting, sort and merge
            team_batting = get_full_batting_stats(team)
            if team_batting.shape[0] == 0:
                print("{} batting was empty".format(team))
                continue
            team_batting = team_batting.loc[
                team_batting['gameId'].isin(
                    list(set(df_base_curr['gameId']))
                ),
            :]
            team_batting = batter_appearance_freq(team_batting, top_batter_count)
            team_batting.sort_values(
                by=['gameDate'], ascending=True, inplace=True
            )
            team_batting.drop(labels=['gameDate'],
                              axis=1, inplace=True)
            team_batting = pivot_stats_wide(team_batting,
                                            swing_col='batterId',
                                            metric_cols=batter_metrics)
            team_batting.rename(columns={'gameId': 'gameId_merge'}, inplace=True)
            df_base_curr = pd.merge(
                df_base_curr, team_batting,
                how='left',
                left_on=['prev_gameid_merge_key'],
                right_on=['gameId_merge'],
                validate='1:1'
            )
            df_base_curr.drop(labels=['gameId_merge'], axis=1, inplace=True)
            
            # Filter to team games for pitching
            team_pitching = get_full_pitching_stats(team)
            if team_pitching.shape[0] == 0:
                print("{} pitching was empty".format(team))
                continue
            team_pitching = team_pitching.loc[
                team_pitching['gameId'].isin(
                    list(set(df_base_curr['gameId']))),
            :]
            team_pitching = pitcher_appearance_freq(team_pitching, top_pitcher_count)
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
            team_pitching = pivot_stats_wide(team_pitching,
                                             swing_col='pitcherId',
                                             metric_cols=pitcher_metrics)
            team_pitching.rename(columns={'gameId': 'gameId_merge'},
                                 inplace=True)
            df_base_curr = pd.merge(
                df_base_curr, team_pitching,
                how='left',
                left_on=['prev_gameid_merge_key'],
                right_on=['gameId_merge'],
                validate='1:1'
            )
            df_base_curr.drop(labels=['gameId_merge'], axis=1, inplace=True)

            # Add scores from boxscores
            boxscores = get_full_boxscores(team)
            boxscores = boxscores.loc[
                boxscores['gameId'].isin(
                    list(set(df_base_curr['gameId']))
                ),
            :]
            print(boxscores.columns)
            boxscores.sort_values(
                by=['date'], ascending=True, inplace=True
            )
            boxscores.drop(labels=['date'], axis=1, inplace=True)
            boxscores.rename(columns={'gameId': 'gameId_merge'},
                             inplace=True)
            df_base_curr = pd.merge(
                df_base_curr, boxscores,
                how='left',
                left_on=['prev_gameid_merge_key'],
                right_on=['gameId_merge'],
                validate='1:1'
            )
            df_base_curr.drop(labels=['gameId_merge'], axis=1,
                              inplace=True)
            # Add

            # Create flag
            try:
                df_base_curr.drop(labels=['home_team_flag', 'away_team_flag'],
                                  axis=1,
                                  inplace=True)
            except:
                pass
            df_base_curr.to_parquet(
                CONFIG.get('paths').get('initial_featurespaces') + \
                '{}_{}_initial_featurespace.parquet'.format(
                    str(yr), str(team)
                )
            )

    
