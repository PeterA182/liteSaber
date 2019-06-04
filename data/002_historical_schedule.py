import os
import sys
import datetime as dt
import numpy as np
import pandas as pd


if __name__ == "__main__":

    # Read in raw gameday files and specify gameId col
    matchups = []

    # Base path
    date_path = CONFIG.get('paths').get('raw')
    for date in os.listdir(date_path):

        # Read in
        df = pd.read_parquet(date_path+date+'boxscore.parquet',
                             columns=['game_id', 'away_team_code',
                                      'home_team_code', 'date'])
        matchups.append(df)

    # Concatenate
    df_matchups = pd.concat(objs=matchups, index=0)
    df_matchups.loc[:, 'date'] = pd.to_datetime(df_matchups['date'])
    df_matchups.sort_values(by=['date', 'gameId'], ascending=True,
                            inplace=True)
    df_matchups.loc[:, 'year'] = df_matchups['date'].dt.year
    years = list(pd.Series.unique(df_matchups['year']))
    df_matchups.drop(labels=['year'], axis=1, inplace=True)
    for yr in years:
        df_matchups.loc[df_matchups['date'].dt.year == yr, :].to_csv(
            CONFIG.get('paths').get('final_datasets') +
            "full_data_base.csv",
            index=False
        )
