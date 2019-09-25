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


def k_percentage_metric_pitcher(data, trails):
    """
    """

    # Sort
    data.sort_values(by=['pitcherId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    for trail in trails:
        data['k'] = data.groupby('pitcherId')\
            ['pitcherSO'].rolling(trail).sum().reset_index(drop=True)
        data['bf'] = data.groupby('pitcherId')\
            ['pitcherBF'].rolling(trail).sum().reset_index(drop=True)
        data['pitcherKPercentage_trail{}'.format(trail)] = (
            data['k'] / data['bf']
        )
    return_cols = ['pitcherId', 'gameDate', 'gameId'] + \
        [x for x in data.columns if 'pitcherKPercentage' in x]
    data = data.loc[:, return_cols]
    return data
        

def walk_percentage_metric_pitcher(data, trails):
    """
    """
    
    # Sort
    data.sort_values(by=['pitcherId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    for trail in trails:
        data['numerator'] = data.groupby('pitcherId')\
            ['pitcherBBAllowed'].rolling(trail).sum().reset_index(drop=True)
        data['denominator'] = data.groupby('pitcherId')\
            ['pitcherBF'].rolling(trail).sum().reset_index(drop=True)
        data['pitcherBBPercentage_trail{}'.format(trail)] = (
            data.numerator / data.denominator
        )
    return_cols = ['pitcherId', 'gameDate', 'gameId'] + \
        [x for x in data.columns if 'pitcherBBPercentage' in x]
    data = data.loc[:, return_cols]
    return data


def era_avg_prev_metric(data, trails):
    """
    """
    
    # Sort
    data.sort_values(by=['pitcherId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    for trail in trails:
        data['pitcherERA_trail{}'.format(trail)] = data.groupby('pitcherId')\
            ['pitcherERA'].rolling(trail).mean().reset_index(drop=True)
    return_cols = ['pitcherId', 'gameDate', 'gameId'] + \
        [x for x in data.columns if 'pitcherERA_trail' in x]
    data = data.loc[:, return_cols]
    return data


def era_max_prev_metric(data, trails):
    """
    """

    # Sort
    data.sort_values(by=['pitcherId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    for trail in trails:
        data['pitcherERA_max_trail{}'.format(trail)] = data.groupby('pitcherId')\
            ['pitcherERA'].rolling(trail).max().reset_index(drop=True)
    return_cols = ['pitcherId', 'gameDate', 'gameId'] + \
        [x for x in data.columns if 'pitcherERA_max' in x]
    data = data.loc[:, return_cols]
    return data


def hr_per_batter_faced(data, trails):
    """
    """

    # Sort
    data.sort_values(by=['pitcherId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    for trail in trails:
        data['numerator'] = data.groupby('pitcherId')\
            ['pitcherHR'].rolling(trail).sum().reset_index(drop=True)
        data['denominator'] = data.groupby('pitcherId')\
            ['pitcherBF'].rolling(trail).sum().reset_index(drop=True)
        data['hrPerBF_trail{}'.format(trail)] = (
            data.numerator / data.denominator
        )
    return_cols = ['pitcherId', 'gameDate', 'gameId'] + \
        [x for x in data.columns if 'hrPerBF_trail' in x]
    data = data.loc[:, return_cols]
    return data


def bf_metric(data, trails):
    """
    """

    # Sort
    print(type(data))
    print(data.shape)
    data.sort_values(by=['pitcherId', 'gameDate', 'gameId'],
                     ascending=True,
                     inplace=True)
    print(data.shape)
    print("trails")
    print(trails)
    for trail in trails:
        data['bf_trail{}'.format(trail)] = data.groupby('pitcherId')\
            ['pitcherBF'].rolling(trail).sum().reset_index(drop=True)
    print("return cols")
    return_cols = ['pitcherId', 'gameDate', 'gameId'] + \
        [x for x in data.columns if 'bf_trail' in x]
    data = data.loc[:, return_cols]
    return data


if __name__ == "__main__":

    # -----------
    # Trailing appearances
    trails = [3, 6]
    
    # -----------
    # Years
    min_year = dt.datetime.now().year
    max_year = dt.datetime.now().year

    if min_year == max_year:
        years = [int(min_year)]
    else:
        years = [int(min_year) + i for i in np.arange(min_year, max_year+1, 1)]
    
    years = [2019]

    # -----------
    # Iterate
    for yr in years:
        print("Now Processing Year :: {}".format(str(yr)))

        # Get all paths for year
        bat_paths = [
            CONFIG.get('paths').get('normalized') + date_str + "/pitching.parquet"
            for date_str in os.listdir(CONFIG.get('paths').get('normalized'))
            if str(yr) in date_str
        ]
        bat_paths = [x for x in bat_paths if os.path.isfile(x)]

        # Process and append batting paths
        df = pd.concat(
            objs=[pd.read_parquet(yr_batting_path) for yr_batting_path in bat_paths],
            axis=0
        )
        df.loc[:, 'gameDate'] = df['gameId'].apply(lambda x: x.split("_"))
        df = df.loc[df['gameDate'].apply(lambda x: str(x[1])) == str(yr), :]
        df.loc[:, 'gameDate'] = df['gameDate'].apply(
            lambda x: dt.datetime(
                year=int(str(x[1])),
                month=int(str(x[2])),
                day=int(str(x[3]))
            )
        )

        # Add Date
        df.loc[:, 'gameDate'] = pd.to_datetime(
            df['gameDate'], infer_datetime_format=True
        )

        # Establish base table
        df_master = df.loc[:, [
            'gameId', 'gameDate', 'pitcherId', 'pitcherTeamFlag'
        ]].drop_duplicates(inplace=False)
        
        # Add FIP
        fip = fip_metric(df, trails)
        df_master = pd.merge(
            df_master,
            fip,
            how='left',
            on=['gameId', 'gameDate', 'pitcherId'],
            validate='1:1'
        )

        # Add K Percentage
        k_pct = k_percentage_metric_pitcher(df, trails)
        df_master = pd.merge(
            df_master,
            k_pct,
            how='left',
            on=['gameId', 'gameDate', 'pitcherId'],
            validate='1:1'
        )

        # Add Walk Percentage
        walk_pct = walk_percentage_metric_pitcher(df, trails)
        df_master = pd.merge(
            df_master,
            walk_pct,
            how='left',
            on=['gameId', 'gameDate', 'pitcherId'],
            validate='1:1'
        )

        # Add ERA Trail
        era_prev = era_avg_prev_metric(df, trails)
        df_master = pd.merge(
            df_master,
            era_prev,
            how='left',
            on=['gameId', 'gameDate', 'pitcherId'],
            validate='1:1'
        )


        # Add ERA Max Trail
        era_max_prev = era_max_prev_metric(df, trails)
        df_master = pd.merge(
            df_master,
            era_max_prev,
            how='left',
            on=['gameId', 'gameDate', 'pitcherId'],
            validate='1:1'
        )

        # HR Per BF
        hr_per_bf = hr_per_batter_faced(df, trails)
        df_master = pd.merge(
            df_master,
            hr_per_bf,
            how='left',
            on=['gameId', 'gameDate', 'pitcherId'],
            validate='1:1'
        )

        # BF Metric
        print(df.shape)
        bf_metric = bf_metric(df, trails)
        df_master = pd.merge(
            df_master,
            bf_metric,
            how='left',
            on=['gameId', 'gameDate', 'pitcherId'],
            validate='1:1'
        )

        #
        # ------------------------------
        # Final
        print("Iterating over gid now")
        for gid in list(pd.Series.unique(df_master.gameId)):
            dest_path = CONFIG.get('paths').get('pitcher_saber') + str(gid) + "/"
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            curr_master = df_master.loc[df_master['gameId'] == gid, :]
            try:
                print("Saving File :: {}".format(str(gid)))
                curr_master.to_csv(
                    CONFIG.get('paths').get('pitcher_saber') + \
                    '{}/pitcher_saber.csv'.format(str(gid)), index=False
                )
            except:
                pass
            try:
                curr_master.to_parquet(
                    CONFIG.get('paths').get('pitcher_saber') + \
                    '{}/pitcher_saber.parquet'.format(str(gid))
                )
            except:
                pass
            
        

            



        
