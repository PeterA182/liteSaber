import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
import yaml
with open(
    "/Users/peteraltamura/Documents/GitHub/liteSaber/data/configuration.yaml",
        "rb"
) as c:
    CONFIG = yaml.load(c)
pd.options.display.max_columns = None


def load_inning_scores(min_date, max_date):
    """
    """

    str_min_date = min_date.strftime("%Y_%m_%d")
    str_max_date = max_date.strftime("%Y_%m_%d")
    
    # Load innings
    df = pd.concat(
        objs=[
            pd.read_parquet(
                CONFIG.get('paths').get('normalized') +
                dd + "/innings.parquet",
                columns=[
                    'gameId', 'gameDate', 'inning_num',
                    'atbat_away_team_runs', 'atbat_home_team_runs'
                ]
            ) for dd in os.listdir(CONFIG.get('paths').get('normalized'))],
        axis=1
    )
    df = df.loc[df['inning_num'].astype(str) == '5', :]
    df = df.loc[(
        (df['gameDate'] >= min_date)
        &
        (df['gameDate'] <= max_date)
    ), :]
    df.drop_duplicates(inplace=True)
    df['atbat_home_team_runs'] = max(df['atbat_home_team_runs'])
    df['atbat_away_team_runs'] = max(df['atbat_away_team_runs'])
    df = df.loc[:, ['gameId', 'gameDate', 'atbat_home_team_runs',
                    'atbat_away_team_runs']].drop_duplicates(
                        inplace=False
                    )
    return df

                                   
def load_starter_metrics(yr):
    """
    """

    # Prep Dates
    str_min_date = min_date.strftime("%Y_%m_%d")
    str_max_date = max_date.strftime("%Y_%m_%d")

    # Load innings
    df = pd.concat(
        objs=[
            pd.read_parquet(
                CONFIG.get('paths').get('normalized') +
                dd + "/innings.parquet",
                columns=[
                    'gameId', 'gameDate', 'inning_num', 'atbat_pitcher',
                    'atbat_batter', 'atbat_away_team_runs', 'atbat_stand',
                    'atbat_home_team_runs', 'atbat_event',
                    'inning_away_team', 'inning_home_team',
                    'atbat_p_throws', 'home_starting_pitcher',
                    'away_starting_pitcher', 'inning_half'
                ]
            ) for dd in os.listdir(CONFIG.get('paths').get('normalized')) if
            str(yr) in str(dd)
        ],
        axis=1
    )
    df = df.loc[df['inning_num'].astype(int) <= 5, :]

    # Return table
    df_metrics = df.loc[:, [
        'gameId', 'gameDate', 'inning_away_team', 'inning_home_team',
        'away_starting_pitcher', 'home_starting_pitcher'
    ]].drop_duplicates(inplace=False) 
    
    # Proportion of innings getting to 5th inning
    df.loc[:, 'home_reliever_by_5th'] = (
        (df['inning_num'].astype(str) == '5')
        &
        (df['home_starting_pitcher'] != df['atbat_pitcher'])
        &
        (df['inning_half'] == 'bottom')
    ).astype(int)
    df.loc[:, 'away_reliver_by_5th'] = (
        (df['inning_num'].astype(str) == '5')
        &
        (df['away_starting_pitcher'] != df['atbat_pitcher'])
        &
        (df['inning_half'] == 'top')
    ).astype(int)
    df_m_add = df.groupby(
        by=['gameId', 'gameDate'],
        as_index=False
    ).agg({'home_reliever_by_5th': 'max',
           'away_reliever_by_5th': 'max'})
    df_metrics = pd.merge(
        df_metrics, df_m_add,
        how='left',
        on=['gameId', 'gameDate'],
        validate='1:1'
    )
    
    # Rolling proportion of games getting to 5th inning for starter
    # this year
    

    # Big inning tendency for pitcher

    # Big inning tendency for hitting team

    # Starter record vs team

    # Starter average inning count vs team

    # Starter average runs by 5th inning bs team

    # Game divisional

    # Historic mathcup count pitcher and team
    
    

    
    
    



if __name__ == "__main__":

    # Configuration

    # Study Period
    min_date = dt.datetime(year=2016, month=a, day=1)
    max_date = dt.datetime(year=2019, month=9, day=23)

    # Base table from Boxscore
    df_base = load_inning_scores(min_date, max_date)

    # Add current season starter metrics
    df_starter_curr_yr = load_starter_metrics(int(max_date.year))
    
