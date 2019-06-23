import os
import gc
import sys
import numpy as np
import pandas as pd
import datetime as dt
import utilities as util
CONFIG = util.load_config()


def get_assembled_team_data(year):
    """
    """

    #
    filenames = [
        x for x in os.listdir(
            CONFIG.get('paths').get('initial_featurespaces')
        ) if str(x[:4]) in year
    ]
    df_base_all = pd.concat(
        objs=[pd.read_parquet(CONFIG.get('paths').get('initial_featurespaces') +
                              fname) for fname in filenames],
        axis=0
    )
    return df_base_all


if __name__ == "__main__":

    # Define years
    min_year = 2017
    max_year = 2017

    # Determine min fill
    min_fill_rate = 0.97
    
    # Calculate years
    years = [
        str(y) for y in np.arange(min_year, max_year+1, 1)
    ]

    # Iterate over years
    for yr in years:
        print("Now Processing Year:: {}".format(str(yr)))

        # Read in all prepped data
        df_base_all = get_assembled_team_data(yr)

        # Prep names of all columns in df_base_all

        # Note columns that will be doubled in merge

        # Iterate over team

        # Label 
