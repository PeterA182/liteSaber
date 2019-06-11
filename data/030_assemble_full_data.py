import os
import gc
import sys
import numpy as np
import pandas as pd
import datetime as dt
import utilities as util
CONFIG = util.load_config()


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
        #df = df.loc[df['gameId'].str.contains(team), :]
        if pd.Series.nunique(df['gameId']) != 1:
            continue
        gameId = df['gameId'].iloc[0]
        df.loc[df['gameId'].apply(lambda x: x.split("_")[4]) == team, 'batterTeamFlag'] = 'away'
        df.loc[df['gameId'].apply(lambda x: x.split("_")[5]) == team, 'batterTeamFlag'] = 'home'
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
        idx_cols = ['gameId', 'batterTeamFlag']
    elif swing_col == 'pitcherId':
        metric_cols = [
            x for x in data.columns if any(
                y in x for y in [
                    'pitcher{}'.format(xx)
                    for xx in metric_cols
                ]
            )
        ]
        idx_cols = ['gameId', 'pitcherTeamFlag']
    else:
        raise

    # Pivot Table
    player_ids = list(set(data[swing_col]))
    data = data.pivot_table(
        index=idx_cols,
        columns=[swing_col],
        values=metric_cols
    )
    data.reset_index(inplace=True)
    data.columns = [
        x[0] if x[1] == '' else x[0]+"_"+str(player_ids.index(x[1]))
        for x in data.columns
    ]
    
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
    for yr in ['2017', '2018']:

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
        #team_prev_merge_key_dict = get_merge_key_dict(df_base)
        prev_game_ids = get_prev_game_id(df_base)

        # Win Loss by Disposition
        home_team_win_pct_home, away_team_win_pct_away = win_loss_by_disposition(df_base)
        
        #
        # ---------------------------------------------
        # PREPARATION TO ENSURE ALL MERGES ARE COMPLETE

        # ---------------------------------------------
        # Iterate over teams and apend final table to list
        complete_team_tables = []
        
        #for team in teams_list:
        for team in teams_list:
        #for team in ['col']:
            if team in ['aas', 'nas', 'umi', 'atf']:
                continue

            print("Now creating featurespace for team: {}".format(team))

            # Subset base for current team
            df_base_curr = df_base.loc[df_base['gameId'].str.contains(team), :]
            print(sum(df_base_curr.gameId.isnull()))

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
            print(sum(df_base_curr.gameId.isnull()))
            if 'gameId_x' in df_base_curr.columns:
                sdfjsdk
            
            # Add prev gameIds to away
            df_base_curr = pd.merge(
                df_base_curr,
                prev_game_ids,
                how='left',
                left_on=['gameId', 'away_code'],
                right_on=['gameId', 'team_code'],
                validate='1:1'
            )
            print(sum(df_base_curr.gameId.isnull()))
            df_base_curr.rename(
                columns={'prevGameId': 'awayPrevGameId'},
                inplace=True
            )
            df_base_curr.drop_duplicates(subset=['homePrevGameId'], inplace=True)
            #HANDLE THS MERGE
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
            print(sum(df_base_curr.gameId.isnull()))
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
            df_base_curr[[
                'gameId', 'awayPrevGameId', 'homePrevGameId',
                'away_code', 'home_code',
                'home_team_win_pct_at_home', 'away_team_win_pct_at_away'
            ]].to_csv('/Users/peteraltamura/Desktop/df_base_curr.csv')

            # --------------------
            # Add Full Linescore to summary
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

            # --------------------
            # Filter to team games for batting, sort and merge
            team_batting = get_full_batting_stats(team)

            # Filter or go to next
            if team_batting.shape[0] == 0:
                print("{} batting was empty".format(team))
                continue
            team_batting = team_batting.loc[
                team_batting['gameId'].isin(
                    list(set(df_base_curr['gameId']))
                ),
            :]

            # Get batter frequency
            team_batting = batter_appearance_freq(team_batting, top_batter_count)
            team_batting.to_csv('/Users/peteraltamura/Desktop/team_batting.csv', index=False)
            team_batting = pivot_stats_wide(team_batting,
                                            swing_col='batterId',
                                            metric_cols=batter_metrics)

            # Split batting stats to home team
            df_base_curr_home = df_base_curr.loc[df_base_curr['home_code'] == team, :]
            team_batting_home = team_batting.loc[team_batting['batterTeamFlag'] == 'home', :]
            team_batting_home.rename(columns={'gameId': 'homeGameId'}, inplace=True)
            df_base_curr_home = pd.merge(
                df_base_curr_home,
                team_batting_home,
                how='left',
                left_on=['homePrevGameId'],
                right_on=['homeGameId'],
                validate='1:1'
            )
            df_base_curr_home.drop(labels=['homeGameId'], axis=1, inplace=True)
            
            # Split batting stats to away team
            df_base_curr_away = df_base_curr.loc[df_base_curr['away_code'] == team, :]
            team_batting_away = team_batting.loc[team_batting['batterTeamFlag'] == 'away', :]
            team_batting_away.rename(columns={'gameId': 'awayGameId'}, inplace=True)
            df_base_curr_away = pd.merge(
                df_base_curr_away,
                team_batting_away,
                how='left',
                left_on=['awayPrevGameId'],
                right_on=['awayGameId'],
                validate='1:1'
            )
            df_base_curr_away.drop(labels=['awayGameId'], axis=1, inplace=True)

            # Concatenate home and away
            df_base_curr = pd.concat(
                objs=[df_base_curr_home, df_base_curr_away],
                axis=0
            )
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

            # Split pitching stats to home team
            df_base_curr_home = df_base_curr.loc[df_base_curr['home_code'] == team, :]
            team_pitching_home = team_pitching.loc[team_pitching['pitcherTeamFlag'] == 'home', :]
            team_pitching_home.rename(columns={'gameId': 'homeGameId'}, inplace=True)
            df_base_curr_home = pd.merge(
                df_base_curr_home,
                team_pitching_home,
                how='left',
                left_on=['homePrevGameId'],
                right_on=['homeGameId'],
                validate='1:1'
            )
            df_base_curr_home.drop(labels=['homeGameId'], axis=1, inplace=True)

            # Split pitching stats to away team
            df_base_curr_away = df_base_curr.loc[df_base_curr['away_code'] == team, :]
            team_pitching_away = team_pitching.loc[team_pitching['pitcherTeamFlag'] == 'away', :]
            team_pitching_away.rename(columns={'gameId': 'awayGameId'}, inplace=True)
            df_base_curr_away = pd.merge(
                df_base_curr_away,
                team_pitching_away,
                how='left',
                left_on=['awayPrevGameId'],
                right_on=['awayGameId'],
                validate='1:1'
            )
            df_base_curr_away.drop(labels=['awayGameId'], axis=1, inplace=True)

            # Concatenate home and away
            df_base_curr = pd.concat(
                objs=[df_base_curr_home, df_base_curr_away],
                axis=0
            )

            # Create flag
            try:
                df_base_curr.drop(labels=['home_team_flag', 'away_team_flag'],
                                  axis=1,
                                  inplace=True)
            except:
                pass

            final_columns = ['gameId']
            final_columns = [
                x for x in df_base_curr.columns if any(
                    y in x for y in batter_metrics
                )
            ]
            final_columns = [
                x for x in df_base_curr.columns if any(
                    y in x for y in pitcher_metrics
                )
            ]
            final_columns.extend([
                'home_team_win_pct_at_home', 'away_team_win_pct_at_away'
            ])
            final_columns.extend(['home_team_winner'])
            df_base_curr[final_columns].to_parquet(
                CONFIG.get('paths').get('initial_featurespaces') + \
                '{}_{}_initial_featurespace.parquet'.format(
                    str(yr), str(team)
                )
            )
            if team == 'col':
                df_base_curr[final_columns].to_csv(
                    '/Users/peteraltamura/Desktop/finished_nya.csv',
                    index=False
                )

    
