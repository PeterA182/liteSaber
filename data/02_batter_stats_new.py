import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
import utilities as util
import multiprocessing as mp
CONFIG = util.load_config()


def walk_percentage_metric(data, trails):
    """
    """

    # Sort
    data.sort_values(by=['batterId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    for trail in trails:
        data['numerator'] = data.groupby('batterId')\
            ['batterBB'].rolling(trail).sum().reset_index(drop=True)
        data['denominator'] = data.groupby('batterId')\
            ['batterAB'].rolling(trail).sum().reset_index(drop=True)
        data['batterWalkPercentage_trail{}'.format(str(trail))] = (
            data.numerator / data.denominator
        )
    return_cols = ['batterId', 'gameDate', 'gameId'] + \
             [x for x in data.columns if 'batterWalkPercentage' in x]
    data = data.loc[:, return_cols]
    return data


def k_percentage_metric(data, trails):
    """
    """
    
    # Sort
    data.sort_values(by=['batterId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    for trail in trails:
        data['numerator'] = data.groupby('batterId')\
            ['batterStrikeOuts'].rolling(trail).sum().reset_index(drop=True)
        data['denominator'] = data.groupby('batterId')\
            ['batterAB'].rolling(trail).sum().reset_index(drop=True)
        data['batterKPercentage_trail{}'.format(str(trail))] = (
            data.numerator / data.denominator
        )
    return_cols = ['batterId', 'gameDate', 'gameId'] + \
             [x for x in data.columns if 'batterKPercentage' in x]
    data = data.loc[:, return_cols]
    return data


def iso_metric(data, trails):
    """
    """

    # Sort
    data.sort_values(by=['batterId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    for trail in trails:
        data['numerator'] = data.groupby('batterId')\
            ['batterSluggingPct'].rolling(trail).mean().reset_index(drop=True)
        data['denominator'] = data.groupby('batterId')\
            ['batterAvg'].rolling(trail).mean().reset_index(drop=True)
        data['batterISO_trail{}'.format(trail)] = (
            data.numerator - data.denominator
        )
    return_cols = ['batterId', 'gameDate', 'gameId'] + \
             [x for x in data.columns if 'batterISO' in x]
    data = data.loc[:, return_cols]
    return data

    
def babip_metric(data, trails):
    """
    BABIP = (H – HR)/(AB – K – HR + SF)
    """

    # Sort
    data.sort_values(by=['batterId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    for trail in trails:
        # ballsinplay
        # hits
        data['hits'] = data.groupby('batterId')\
            ['batterH'].rolling(trail).sum().reset_index(drop=True)
        data['doubles'] = data.groupby('batterId')\
            ['batterDoubles'].rolling(trail).sum().reset_index(drop=True)
        data['triples'] = data.groupby('batterId')\
            ['batterTriples'].rolling(trail).sum().reset_index(drop=True)
        data['bip'] = (data['hits'] + data['doubles'] + data['triples'])
        # Home RUns
        data['hr'] = data.groupby('batterId')\
            ['batterHR'].rolling(trail).sum().reset_index(drop=True)
        # AB
        data['ab'] = data.groupby('batterId')\
            ['batterAB'].rolling(trail).sum().reset_index(drop=True)
        # K
        data['k'] = data.groupby('batterId')\
            ['batterStrikeOuts'].rolling(trail).sum().reset_index(drop=True)
        # Sac Fly
        data['sacfly'] = data.groupby('batterId')\
            ['batterSacFlys'].rolling(trail).sum().reset_index(drop=True)
        # Sac Bunt
        data['sacBunt'] = data.groupby('batterId')\
            ['batterSacBunts'].rolling(trail).sum().reset_index(drop=True)
        # Calculate
        data['batterBABIP_trail{}'.format(trail)] = (
            (data['bip']-data['hr'])/
            (data['ab']-data['k']-data['hr']-data['sacBunt']+data['sacfly'])
        )
    return_cols = ['batterId', 'gameDate', 'gameId'] + \
             [x for x in data.columns if 'batterBABIP' in x]
    data = data.loc[:, return_cols]
    return data


def woba_metric(data, trails):
    """
    wOBA = (0.690×uBB + 0.722×HBP + 0.888×1B + 1.271×2B + 1.616×3B +
        2.101×HR) / (AB + BB – IBB + SF + HBP)
    """

    # Sort
    data.sort_values(by=['batterId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    for trail in trails:
        # BB
        data['bb'] = data.groupby('batterId')\
            ['batterBB'].rolling(trail).sum().reset_index(drop=True)
        # HBP
        data['hbp'] = data.groupby('batterId')\
            ['batterHBP'].rolling(trail).sum().reset_index(drop=True)
        # single
        data['singles'] = data.groupby('batterId')\
            ['batterH'].rolling(trail).sum().reset_index(drop=True)
        # double
        data['doubles'] = data.groupby('batterId')\
            ['batterDoubles'].rolling(trail).sum().reset_index(drop=True)
        # triple
        data['triples'] = data.groupby('batterId')\
            ['batterTriples'].rolling(trail).sum().reset_index(drop=True)
        # home run
        data['hr'] = data.groupby('batterId')\
            ['batterHR'].rolling(trail).sum().reset_index(drop=True)
        # ab
        data['ab'] = data.groupby('batterId')\
            ['batterAB'].rolling(trail).sum().reset_index(drop=True)
        # bb
        data['bb'] = data.groupby('batterId')\
            ['batterBB'].rolling(trail).sum().reset_index(drop=True)
        # sac fly
        data['sacfly'] = data.groupby('batterId')\
            ['batterSacFlys'].rolling(trail).sum().reset_index(drop=True)
        # hbp
        data['hbp'] = data.groupby('batterId')\
            ['batterHBP'].rolling(trail).sum().reset_index(drop=True)
        
        data['woba_trail{}'.format(trail)] = (
            (0.690*data['bb'] + 0.722*data['hbp'] + 0.888*data['singles'] + \
             1.271*data['doubles'] + 1.616*data['triples'] + 2.101*data['hr']) /
            (data['ab']+data['bb']+data['sacfly']+data['hbp'])
        )
    return_cols = ['batterId', 'gameDate', 'gameId'] + \
             [x for x in data.columns if 'woba_trail' in x]
    data = data.loc[:, return_cols]
    return data





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

        # Establish base table
        df_master = df.loc[:, [
            'gameId', 'gameDate', 'batterId', 'batterTeamFlag'
        ]].drop_duplicates(inplace=False)
        
        # Add Walk Percentage
        walk_pct = walk_percentage_metric(df, trails)
        df_master = pd.merge(
            df_master,
            walk_pct,
            how='left',
            on=['gameId', 'gameDate', 'batterId'],
            validate='1:1'
        )

        # Add K Pct
        k_pct = k_percentage_metric(df, trails)
        df_master = pd.merge(
            df_master,
            k_pct,
            how='left',
            on=['gameId', 'gameDate', 'batterId'],
            validate='1:1'
        )
        
        # ISO
        iso = iso_metric(df, trails)
        df_master = pd.merge(
            df_master,
            iso,
            how='left',
            on=['gameId', 'gameDate', 'batterId'],
            validate='1:1'
        )

        # BABIP
        babip = babip_metric(df, trails)
        df_master = pd.merge(
            df_master,
            babip,
            how='left',
            on=['gameId', 'gameDate', 'batterId'],
            validate='1:1'
        )

        # WOBA
        woba = woba_metric(df, trails)
        df_master = pd.merge(
            df_master,
            woba,
            how='left',
            on=['gameId', 'gameDate', 'batterId'],
            validate='1:1'
        )

        #
        # ------------------------------
        # Final
        for gid in list(pd.Series.unique(df_master.gameId)):
            dest_path = CONFIG.get('paths').get('batter_saber') + str(gid) + "/"
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            df_master.loc[df_master['gameId'] == gid, :].to_parquet(
                CONFIG.get('paths').get('batter_saber') + \
                '{}batter_saber.parquet'.format(str(gid))
            )
            df_master.loc[df_master['gameId'] == gid, :].to_csv(
                CONFIG.get('paths').get('batter_saber') + \
                '{}batter_saber.csv'.format(str(gid)), index=False
            )
