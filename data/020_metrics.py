import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import utilities as util
CONFIG = util.load_config()


def process_date_games(path):
    """
    """

    #


    return 0


if __name__ == "__main__":

    # Run Log
    min_date = dt.datetime(year=2018, month=8, day=1)
    max_date = dt.datetime(year=2018, month=8, day=2)

    # Iterate over years
    years = [y for y in np.arange(min_date.year, max_date.year+1, 1)]
    dates = [min_date+dt.timedelta(days=i)
             for i in range((max_date-min_date).days+1)]

    # Establish path
    path = CONFIG.get('paths').get('normalized')
    for dd in dates:
        dd = dd.strftime('year_%Ymonth_%mday_%d/')
        path += dd         
        print(path)
        process_date_games(path)
