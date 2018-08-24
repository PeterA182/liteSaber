import sys
import os

import pandas as pd
import numpy as np
import datetime as dt
from utilities import methods as utils
from utilities.mappings import maps as colmaps
from lifelines import KaplanMeierFitter, CoxPHFitter
pd.set_option("display.max_columns", 500)

#
# Config
CONFIG = {
    'inpath': "/Volumes/Transcend/gameday/",
    'windowStartDate': dt.datetime(year=2016, month=1, day=1),
    'windowEndDate': dt.datetime(year=2016, month=12, day=1)
}


if __name__ == "__main__":

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

    # Sort by Player and Game
    df_batting.sort_values(by=['game_id', 'id'], ascending=True, inplace=True)

    # Rename
    df_batting.columns = [
        colmaps['batting'][x] if x in colmaps['batting'].keys()
        else x for x in df_batting.columns]

    # Number games
    df_batting['game_number'] = df_batting.groupby(by=['id'])['hit'].\
            rank(method='first', ascending=True)

    # Flag Slumps
    print(df_batting[['hit', 'id']].head(10))
    slump_msk = ((df_batting['hit'] == 0) &
                 (df_batting['hit'].shift(1) == 0) &
                 (df_batting['hit'].shift(2) == 0))
    df_batting['slump'] = slump_msk.astype(float)


    # KMF
    batting_model = df_batting.drop(
        labels=['game_id', 'name', 'name_display_first_last', 'note',
                'pos', 'team_team_flag'],
        axis=1,
        inplace=False
    )
    # Handle nulls
    for col in batting_model.columns:
        if sum(batting_model[col].isnull()) > 0:
            batting_model.loc[batting_model[col].isnull(), col] = \
                np.nanmedian(batting_model[col])
    cph = CoxPHFitter()
    cph.fit(batting_model,
            duration_col='game_number',
            event_col='slump',
            show_progress=True)
    print(cph.score_)










