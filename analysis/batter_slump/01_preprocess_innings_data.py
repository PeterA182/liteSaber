import sys
import os

import pandas as pd
import numpy as np
import datetime as dt
from utilities import methods as utils
from utilities.mappings import maps as colmaps
from lifelines import KaplanMeierFitter, CoxPHFitter
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 200)

#
# Config
CONFIG = {
    'inpath': "/Volumes/Transcend/gameday/",
    'windowStartDate': dt.datetime(year=2015, month=4, day=1),
    'windowEndDate': dt.datetime(year=2015, month=11, day=1),
    'local': False
}


if __name__ == "__main__":

    # Read in
    if CONFIG.get('local'):
        df_innings = pd.read_csv('/Users/peteraltamura/Desktop/innings_sample.csv')
    else:
        subdirs = utils.get_gameday_subdirs(path=CONFIG.get('inpath'),
                                      window_start=CONFIG.get('windowStartDate'),
                                      window_end=CONFIG.get('windowEndDate'),
                                      left_incl=True, right_incl=True)

        # Append Files
        df_innings = pd.concat(
            objs=[pd.read_parquet(CONFIG.get('inpath') + subdir + '/innings.parquet')
                  for subdir in subdirs if
                  "innings.parquet" in os.listdir(CONFIG.get('inpath')+subdir)],
            axis=0
        )
    df_dtypes = df_innings.dtypes.reset_index()
    df_dtypes.columns = ['Variable', 'DType']

    # Rename
    print(df_dtypes)

    # Sort by Player and Game
    df_innings.sort_values(by=['game_id', 'inning_num', 'atbat_num'],
                           ascending=True,
                           inplace=True)

    # Get batter specific table




    """
    cph = CoxPHFitter()
    cph.fit(batting_model,
            duration_col='game_number',
            event_col='slump',
            show_progress=True)
    print(cph.score_)
    """









