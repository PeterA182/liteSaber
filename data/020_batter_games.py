import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
import utilities as util
import multiprocessing as mp
CONFIG = util.load_config()

if __name__ == "__main__":

    # ----------
    # Years
    min_year = 2017
    max_year = 2017
    if min_year == max_year:
        years = [min_year]
    else:
        years = [max_year]

    # Process
    for yr in years:

        # Get all paths for year
        bat_paths = [
            CONFIG.get('paths').get('normalized') + date_str + "/batting.parquet"
            for date_str in os.listdir(CONFIG.get('paths').get('normalized'))
        ]
        bat_paths = [x for x in bat_paths if os.path.isfile(x)]

        # Process and append batting paths
        df = pd.concat(
            objs=[pd.read_parquet(yr_batting_path) for yr_batting_path in bat_paths],
            axis=0
        )

        # Add Date
        df.loc[:, 'gameDate'] = pd.to_datetime(
            df['gameDate'], infer_datetime_format=True
        )

        df_master = df.loc[:, [
            'gameId', 'gameDate', 'batterId', 'batterTeamFlag'
        ]].drop_duplicates(inplace=False)

        # Base Table
        stats = [
            'batterAssists', 'batterAB', 'batterFO', 'batterAvg', 'batterBB',
            'batterH', 'batterHBP', 'batterHR', 'batterDoubles',
            'batterErrors', 'batterGroundOuts', 'batterLob', 'batterOBP',
            'batterOPS', 'batterR', 'batterRBI', 'batterSluggingPct',
            'batterStrikeOuts', 'batterTriples' 
        ]

        # Get prev 3, 5, 10 batting stats
        df.sort_values(by=['batterId', 'gameId'], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        for days in [3, 5, 10]:
            for stat in stats:
                df['{}_trail_{}_mean'.format(stat, str(days))] = df.groupby('batterId')\
                    [stat].rolling(days).mean().reset_index(drop=True)
                df['{}_trail_{}_max'.format(stat, str(days))] = df.groupby('batterId')\
                    [stat].rolling(days).max().reset_index(drop=True)
                df['{}_trail_{}_min'.format(stat, str(days))] = df.groupby('batterId')\
                    [stat].rolling(days).min().reset_index(drop=True)
                df['{}_trail_{}_var'.format(stat, str(days))] = df.groupby('batterId')\
                    [stat].rolling(days).var().reset_index(drop=True)
                df_master = pd.merge(
                    df_master,
                    df[['batterId', 'gameId', '{}_trail_{}_mean'.format(stat, str(days)),
                        '{}_trail_{}_max'.format(stat, str(days)), '{}_trail_{}_min'.format(stat, str(days)),
                        '{}_trail_{}_var'.format(stat, str(days))]],
                    how='left',
                    on=['gameId', 'batterId'],
                    validate='1:1'
                )

        # Get season so far
        for stat in stats:
            df.sort_values(by=['batterId', 'gameId'], ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df['{}_trail_season'.format(stat)] = df.groupby('batterId')\
                [stat].cumsum().reset_index(drop=True)
            df_master = pd.merge(
                df_master,
                df[['batterId', 'gameId', '{}_trail_season'.format(stat)]],
                how='left',
                on=['gameId', 'batterId'],
                validate='1:1'
            )
        
        # Final
        for gid in list(pd.Series.unique(df_master.gameId)):
            dest_path = CONFIG.get('paths').get('batter_stats') + str(gid) + "/"
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            df_master.loc[df_master['gameId'] == gid, :].to_parquet(
                CONFIG.get('paths').get('batter_stats') + \
                '{}batter_stats.parquet'.format(str(gid))
            )
            df_master.loc[df_master['gameId'] == gid, :].to_csv(
                CONFIG.get('paths').get('batter_stats') + \
                '{}batter_stats.csv'.format(str(gid)), index=False
            )
            
