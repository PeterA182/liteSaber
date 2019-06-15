import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
import utilities as util
import multiprocessing as mp
CONFIG = util.load_config()


def fip_metric(data, trails):
    """
    FIP = ((13*HR)+(3*(BB+HBP))-(2*K))/IP + constant
    """

    # Add FIP Constant
    data = util.add_fip_constant(data)
    
    # Sort
    data.sort_values(by=['pitcherId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    for trail in trails:
        data['hr'] = data.groupby('pitcherId')\
            ['pitcherHR'].rolling(trail).sum().reset_index(drop=True)
        # bb and hbp together
        data['bb'] = data.groupby('pitcherId')\
            ['pitcherBBAllowed'].rolling(trail).sum().reset_index(drop=True)
        data['k'] = data.groupby('pitcherId')\
            ['pitcherSO'].rolling(trail).sum().reset_index(drop=True)
        # Use ip estimate as stand in for innings from battersfaced
        data['bf'] = data.groupby('pitcherId')\
            ['pitcherBF'].rolling(trail).sum().reset_index(drop=True)
        data['fip_trail{}'.format(trail)] = (
            (((13*data['hr'])+(3*data['bb']))-(2*data['k'])) /
            (data['bf']+data['cFIP'])
        )
    return_cols = ['pitcherId', 'gameDate', 'gameId'] + \
        [x for x in data.columns if 'fip_trail' in x]
    data = data.loc[:, return_cols]
    return data


def k_pct(data, trails):
    """
    """

    # Sort
    data.sort_values(by=['pitcherId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    for trail in trails:
        data['



if __name__ == "__main__":

    # -----------
    # Trailing appearances
    trails = [3, 6]
    
    # -----------
    # Years
    min_year = 2017
    max_year = 2017
    if min_year == max_year:
        years = [min_year]
    else:
        years = [min_year + i for i in np.arange(min_year, max_year+1, 1)]

    # -----------
    # Iterate
    for yr in years:

        # Get all paths for year
        bat_paths = [
            CONFIG.get('paths').get('normalized') + date_str + "/pitching.parquet"
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

        # Establish base table
        df_master = df.loc[:, [
            'gameId', 'gameDate', 'pitcherId', 'pitcherTeamFlag'
        ]].drop_duplicates(inplace=False)
        
        # Add Walk Percentage
        fip = fip_metric(df, trails)
        df_master = pd.merge(
            df_master,
            fip,
            how='left',
            on=['gameId', 'gameDate', 'pitcherId'],
            validate='1:1'
        )

            



        