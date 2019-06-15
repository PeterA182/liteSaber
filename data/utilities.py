import sys
import os
import yaml
import pandas as pd


def load_config():
    with open("configuration.yaml", "rb") as ff:
        config = yaml.load(ff)
    return config


def add_fip_constant(data):
    """
    """

    CONFIG = load_config()

    # Add year if year not in data
    if 'gameYear' not in data.columns:
        data['year'] = data['gameId'].apply(lambda x: x.split("_")[1]).astype(str)

    # Read in constants
    woba_fip_const = pd.read_csv(
        CONFIG.get('paths').get('reference') + \
        CONFIG.get('reference_files').get('woba_fip'),
        dtype=str
    )
    woba_fip_const = woba_fip_const.loc[:, ['Season', 'cFIP']]
    woba_fip_const.loc[:, 'cFIP'] = woba_fip_const['cFIP'].astype(float)
    woba_fip_const.rename(columns={'Season': 'year'}, inplace=True)
    data = pd.merge(
        data,
        woba_fip_const,
        how='left',
        on=['year'],
        validate='m:1'
    )
    return data

