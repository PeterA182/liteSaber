import os
import gc
import sys
import pandas as pd
import datetime as dt
import numpy as np
import utilities as util
CONFIG = util.load_config()


def get_matchup_base_table(date):
    """
    """

    # Read in current year Summaries (will be cut down later)
    df_base = pd.read_parquet(
        CONFIG.get('paths').get('raw') + \
        date.strftime("year_%Ymonth_%mday_%d") +
        "/" + "probableStarters.parquet"
    )

    return df_base


if __name__ == "__main__":


    # ----------  ----------  ----------
    # Parameters
    date = dt.datetime(year=2019, month=6, day=30)
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

    # ---------- ---------- ----------
    # Get Basic Matchup Table
    df_matchup_base = get_matchup_base_table(date=date)

    # Narrow to immediate dimensions
    df_matchup_base = df_matchup_base.loc[:, [
        'gameId', 'startingPitcherId', 'probableStarterSide'
    ]]

    # Only keep starting pitcher id that are filled
    df_matchup_base = df_matchup_base.loc[
        df_matchup_base['startingPitcherId'].notnull(), :]

    # Only keep gameIds with both home and away
    vc = df_matchup_base['gameId'].value_counts()
    vc = pd.DataFrame(vc).reset_index(inplace=False)
    vc.columns = ['gameId', 'freq']
    gids = list(vc.loc[vc['freq'] == 2, :]['gameId'])
    df_matchup_base = df_matchup_base.loc[
        df_matchup_base['gameId'].isin(gids), :]

    # Pivot out to flatten
    df_matchup_base = df_matchup_base.pivot_table(
        index=['gameId'],
        columns=['probableStarterSide'],
        values=['startingPitcherId'],
        aggfunc='first'
    )
    df_matchup_base.reset_index(inplace=True)
    df_matchup_base.columns = [
        x[0] if x[1] == '' else x[0]+"_"+x[1] for
        x in df_matchup_base.columns
    ]

    # Rename
    df_matchup_base.rename(
        columns={
            'startingPitcherId_away': 'away_starting_pitcher',
            'startingPitcherId_home': 'home_starting_pitcher'
        },
        inplace=True
    )
    df_matchup_base = df_matchup_base.loc[:, [
        'gameId', 'away_starting_pitcher', 'home_starting_pitcher'
    ]]
    df_matchup_base.to_csv('/Users/peteraltamura/Desktop/df_matchup_base_tomorrow.csv', index=False)
