import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
import utilities as util
CONFIG = util.load_config()


if __name__ == "__main__":
    
    # Read in raw gameday files and specify gameId col
    matchups = []

    # Base path
    date_path = CONFIG.get('paths').get('raw')
    for date in os.listdir(date_path):

        # Read in
        try:
            df = pd.read_parquet(
                date_path+date+'/boxscore.parquet',
                columns=['gameId', 'away_team_code',
                         'home_team_code', 'date']
            )
        except:
            continue
        matchups.append(df)

    # Concatenate
    df_matchups = pd.concat(objs=matchups, axis=0)
    df_matchups.loc[:, 'date'] = pd.to_datetime(df_matchups['date'])
    df_matchups.sort_values(by=['date', 'gameId'], ascending=True,
                            inplace=True)
    df_matchups.loc[:, 'year'] = df_matchups['date'].dt.year
    years = list(pd.Series.unique(df_matchups['year']))
    df_matchups.drop(labels=['year'], axis=1, inplace=True)
    if not os.path.exists(CONFIG.get('paths').get('final_datasets')):
        os.makedirs(CONFIG.get('paths').get('final_datasets'))
    years = [x for x in years if str(x) != 'nan']
    for yr in [x for x in years if str(x) != 'nan']:
        df_matchups.loc[df_matchups['date'].dt.year == yr, :].to_csv(
            CONFIG.get('paths').get('final_datasets') +
            "{}_full_data_base.csv".format(str(yr).replace(".0", "")),
            index=False
        )
        
