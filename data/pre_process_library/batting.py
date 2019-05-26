import os
import sys
import datetime as dt
import pandas as pd
import numpy as np


def hits_in_last_games(data, days_lookback):
    """
    """

    # Sort by batter and gameDate
    data.sort_values(
        by=['batterId', 'gameDate'],
        ascending=True,
        inplace=True
    )

    # Rolling
    data = data.groupby('batterId')['batterH'].apply(
        pd.rolling_sum,
        days_lookback,
        min_periods=days_lookback
    )
    
