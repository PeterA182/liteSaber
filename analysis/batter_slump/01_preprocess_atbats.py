import sys
import os

import pandas as pd
import numpy as np
import datetime as dt
from utilities import methods as utils
from utilities.mappings import maps as colmaps

pd.set_option("display.max_columns", 500)

# Config
CONFIG = {
    'inpath': "/Volumes/Transcend/gameday/",
    'windowStartDate': dt.datetime(year=2016, month=1, day=1),
    'windowEndDate': dt.datetime(year=2016, month=12, day=1)
}


if __name__ == "__main__":

    # batting. boxscore, pitching,

    # Read in
    subdirs = utils.get_gameday_subdirs(path=CONFIG.get('inpath'),
                                  window_start=CONFIG.get('windowStartDate'),
                                  window_end=CONFIG.get('windowEndDate'),
                                  left_incl=True, right_incl=True)

    # Append Files
    df_batting = pd.concat(
        objs=[pd.read_parquet(CONFIG.get('inpath') + subdir + '/batting.parquet')
              for subdir in subdirs if
              "batting.parquet" in os.listdir(CONFIG.get('inpath')+subdir)],
        axis=0
    )

    df_batting.sort_values(by=['game_id', 'id'], ascending=True, inplace=True)

    # Rename
    df_batting.columns = [
        colmaps['batting'][x] if x in colmaps['batting'].keys()
        else x for x in df_batting.columns]
    




