import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import inspect
import utilities as util
from pre_process_library import rename as names
from pre_process_library import additions as add
CONFIG = util.load_config()


def pitcher_id_indicator(data):
    """
    """
    for pid in list(pd.Series.unique(data['pitcherId'])):
        data['ind_pitcherId_{}'.format(str(pid))] = (
            data['pitcherId'] == pid
        ).astype(int)
    return data


def fip_in_last_5_app(data):
    """
    (13*HR)+(3*(BB+HBP))-(2*K)/(IP)
    """

    # Appearance Rankings for each pitcher
    cols = [
        'inning_num',
        'atbat_num',
        'pitch_id' # (NOT NULL)
        'atbat_pitcher',
        'atbat_p_throws', 
    
