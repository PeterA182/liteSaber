import os
import sys
import pandas as pd
import numpy as np
import utilities as util
CONFIG = util.load_config()


def rename_table(data, tablename):
    """
    """

    # Get dictionary
    dict_ = pd.read_csv(
        CONFIG.get('paths').get('reference') + \
        CONFIG.get('filenames').get('data_dict'),
        dtype=str
    )
    rename_dict = dict_.loc[dict_['file'] == tablename, :]
    rename_dict = rename_dict.loc[rename_dict['map'].notnull(), :]
    rename_dict = rename_dict.set_index('col')['map'].to_dict()
    data.rename(
        columns=rename_dict,
        inplace=True
    )
    return data


def remap_dtypes(data, tablename):
    """
    """

    # Get Dictionary
    dict_ = pd.read_csv(
        CONFIG.get('paths').get('reference') + \
        CONFIG.get('filenames').get('data_dict'),
        dtype=str
    )
    dt_dict = dict_.loc[dict_['file'] == tablename, :]
    dt_dict = dt_dict.loc[dt_dict['dtype'].notnull(), :]
    dt_dict = dt_dict.set_index('map')['dtype'].to_dict()
    for col, dt in dt_dict.items():
        data.loc[:, col] = data[col].astype(dt)
    return data
