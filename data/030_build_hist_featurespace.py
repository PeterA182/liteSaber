import os
import gc
import sys
import pandas as pd
import datetime as dt
import numpy as np
import utilities as util
CONFIG = util.load_config()

"""


"""


def get_matchup_base_table(year):
    """
    """

    # Read in current year Summaries (will be cut down later)
    df_base = pd.concat(
        objs=[
            pd.read_parquet(
                CONFIG.get('paths').get('normalized') + \
                dd + "/" + "game_linescore_summary.parquet"
            ) for dd in os.listdir(
                CONFIG.get('paths').get('normalized')
            ) if str(year) in dd and
            os.path.isfile(CONFIG.get('paths').get('normalized') + \
                dd + "/" + "game_linescore_summary.parquet"
            )
        ],
        axis=0
    )
    return df_base


def add_matchup_features(data):
    """
    Convert raw dimensions from linescore base table into 
    proper features and return
    WIN / LOSS 
    """

    # Time of Day - Home / Away
    # Away AM PM Flag
    msk = (data['away_ampm'].str.lower().strip() == 'pm')
    data['away_is_pm'] = msk.astype(float)
    data.drop(labels=['away_ampm'], axis=1, inplace=True)

    # Home AM PM Flag
    msk = (data['home_ampm'].str.lower().strip() == 'pm')
    data['home_is_pm'] = msk.astype(float)
    data.drop(labels=['home_ampm'], axis=1, inplace=True)

    # Away Division
    msk = (data['away_division'].str.lower().strip() == 'w')
    data['away_div_is_west'] = msk.astype(float)
    msk = (data['away_division'].str.lower().strip() == 'e')
    data['away_div_is_east'] = msk.astype(float)
    msk = (data['away_division'].str.lower().strip() == 'c')
    data['away_div_is_central'] = msk.astype(float)
    data.drop(labels=['away_division'], axis=1, inplace=True)

    # Home Division
    msk = (data['home_division'].str.lower().strip() == 'w')
    data['home_div_is_west'] = msk.astype(float)
    msk = (data['home_division'].str.lower().strip() == 'e')
    data['home_div_is_east'] = msk.astype(float)
    msk = (data['home_division'].str.lower().strip() == 'c')
    data['home_div_is_central'] = msk.astype(float)
    data.drop(labels=['home_division'], axis=1, inplace=True)

    # Away Win / Loss Pct Season to Date
    data.loc['away_win_loss_pct_sntd'] = (
        data['away_win'] / (
            data['away_win'] + data['away_loss']
        )
    )
    data.drop(labels=['away_win', 'away_loss'], axis=1, inplace=True)
    data.loc['home_win_loss_pct_sntd'] = (
        data['home_win'] / (
            data['home_win'] + data['home_loss']
        )
    )
    data.drop(labels=['home_win', 'home_loss'], axis=1, inplace=True)

    
    
    
    

if __name__ == "__main__":

    outpath = "/Users/peteraltamura/Desktop/"

    
    # ----------  ----------  ----------
    # Parameters
    year = '2018'
    bullpen_top_pitcher_count = 6
    pitcher_metrics = [
        'BF', 'ER', 'ERA', 'HitsAllowed', 'Holds',
        'SeasonLosses', 'SeasonWins', 'numberPitches',
        'Outs', 'RunsAllowed', 'Strikes', 'SO'
    ]
    top_batter_count = 12
    batter_metrics = [
        'Assists', 'AB', 'BB', 'FO', 'Avg', 'H',
        'HBP', 'HR', 'Doubles' 'GroundOuts', 'batterLob',
        'OBP', 'OPS', 'R', 'RBI', 'SluggingPct',
        'StrikeOuts', 'Triples'
    ]

    # ----------  ----------  ----------
    # Read in Line Score to get basis for each game played
    df_matchup_base = get_matchup_base_table(year)
    df_matchup_base.to_csv(outpath+"df_matchup_base.csv", index=False)

    df_matchup_base = add_matchup_features(df_matchup_base)
    
