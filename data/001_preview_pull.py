import os
import numpy as np
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ETREE
import datetime as dt
pd.set_option('display.max_columns', 500)


def add_pitcher_ids(data):
    """
    """
    last_registries = [
        fname for fname in sorted(os.listdir(ref_dest))[-50:]
    ]
    registry = pd.concat(
        objs=[
            pd.read_parquet(ref_dest + fname) for fname in last_registries
        ],
        axis=0
    )
    for col in ['first_name', 'last_name']:
        registry.loc[:, col] = registry[col].astype(str)
        registry.loc[:, col] = \
            registry[col].apply(lambda x: x.lower().strip())
    registry.to_csv('/Users/peteraltamura/Desktop/registry_all.csv')
    registry.reset_index(drop=True, inplace=True)
    registry.drop_duplicates(
        subset=['first_name', 'last_name', 'team'],
        inplace=True
    )
    data[['starterFirstName', 'starterLastName']].to_csv(
        '/Users/peteraltamura/Desktop/data.csv')
    registry.to_csv('/Users/peteraltamura/Desktop/registry.csv')
    data = pd.merge(
        data,
        registry,
        how='left',
        left_on=['starterFirstName', 'starterLastName', 'team'],
        right_on=['first_name', 'last_name', 'team'],
        validate='1:1'
    )
    return data


def extract_probables(data):
    """
    Extracts probable home and away pitchers from atv_preview.xml
    """

    resp = data.getroot()

    # Home
    try:
        pitcher_prev_name = resp[0][0][0][2][0][1][0].text
    except IndexError as IE:
        pitcher_prev_name = np.NaN
    try:
        pitcher_prev_stat = resp[0][0][0][2][0][1][1].text
    except IndexError as IE:
        pitcher_prev_stat = np.NaN
    try:
        pitcher_prev_side = resp[0][0][0][2][0][1][2].text
    except IndexError as IE:
        pitcher_prev_side = np.NaN
    df_home = pd.DataFrame({'probableStarterName': [pitcher_prev_name],
                       'probableStarterStat': [pitcher_prev_stat],
                       'probableStarterSide': [pitcher_prev_side]})

    # Away
    try:
        pitcher_prev_name = resp[0][0][0][2][1][1][0].text
    except IndexError as IE:
        pitcher_prev_name = np.NaN
    try:
        pitcher_prev_stat = resp[0][0][0][2][1][1][1].text
    except IndexError as IE:
        pitcher_prev_stat = np.NaN
    try:
        pitcher_prev_side = resp[0][0][0][2][1][1][2].text
    except IndexError as IE:
        pitcher_prev_side = np.NaN
    df_away = pd.DataFrame({'probableStarterName': [pitcher_prev_name],
                            'probableStarterStat': [pitcher_prev_stat],
                            'probableStarterSide': [pitcher_prev_side]})
    df = pd.concat(objs=[df_home, df_away], axis=0)
    return df


def scrape_game_previews(date):
    """
    """

    date_url = "year_{}/month_{}/day_{}".format(
        str(date.year).zfill(4),
        str(date.month).zfill(2),
        str(date.day).zfill(2)
    )
    full_url = base_url + date_url
    print(full_url)

    # Generate list of gid links
    test_resp = urllib.request.urlopen(full_url)
    req = BeautifulSoup(test_resp)
    game_links = [x for x in req.find_all('a') if
                  str(x.get('href'))[7:10] == 'gid']

    # Previews
    probable_starters = []
    for gid in game_links:
        print("        {}".format(str(gid)))
        try:
            game_id = str(gid.get('href'))[7:]
            rbs = full_url + "/" + str(gid.get('href'))[7:] + 'atv_preview.xml'
            resp = urllib.request.urlopen(rbs)
            resp = ETREE.parse(resp)
            df = extract_probables(resp)
            df['gameId'] = game_id
            probable_starters.append(df)
        except ValueError as VE:
            pitcher_prev = pd.DataFrame()
            pass
    try:
        probable_starters = pd.concat(
            objs=probable_starters,
            axis=0
        )
    except ValueError as VE:
        probable_starters = pd.DataFrame()
        pass

    # Filter to Games with two probable starters
    probable_starters = probable_starters.loc[
        probable_starters['probableStarterName'].notnull(), :]
    psvc = probable_starters['gameId'].value_counts()
    psvc = pd.DataFrame(psvc).reset_index(inplace=False)
    psvc.columns = ['gameId', 'freq']
    psvc = psvc.loc[psvc['freq'] == 2, :]
    games = list(set(psvc['gameId']))
    probable_starters = probable_starters.loc[probable_starters['gameId'].isin(games), :]
    
    # Add Format Pitcher Name
    # First Name - assign out
    probable_starters['starterFirstName'] =\
        probable_starters['probableStarterName'].apply(
            lambda s: s.split(" ")[0]
        )
    # Format
    probable_starters.loc[:, 'starterFirstName'] = \
        probable_starters['starterFirstName'].apply(
            lambda x: x.lower()
        )
    # Last Name - assign out
    probable_starters['starterLastName'] = \
        probable_starters['probableStarterName'].apply(
            lambda s: s.split(" ")[1]
        )
    # Format
    probable_starters.loc[:, 'starterLastName'] = \
        probable_starters['starterLastName'].apply(
            lambda x: x.lower()
        )
    # Strip both
    for x in ['starterFirstName', 'starterLastName']:
        probable_starters.loc[:, x] = probable_starters[x].str.strip()

    # Add Home Team / Away Team
    probable_starters.loc[:, 'probableStarterSide'] = \
        probable_starters['probableStarterSide'].apply(
            lambda x: x.strip().lower()
        )
    probable_starters['homeTeam'] = probable_starters['gameId'].apply(
        lambda x: x.split("_")[5]
    )
    probable_starters['awayTeam'] = probable_starters['gameId'].apply(
        lambda x: x.split("_")[4]
    )
    probable_starters.reset_index(drop=True, inplace=True)
    probable_starters.loc[
        probable_starters['probableStarterSide'] == 'home',
        'team'] = probable_starters['homeTeam'].str[:3]
    probable_starters.loc[
        probable_starters['probableStarterSide'] == 'away',
        'team'] = probable_starters['awayTeam'].str[:3]
                          
    # Add Pitcher ID From team register
    probable_starters = add_pitcher_ids(probable_starters)
    probable_starters.rename(
        columns={
            'id': 'startingPitcherId',
            'team': 'startingPitcherTeam',
            'dob': 'startingPitcherDob',
            'throws': 'startingPitcherThrows',
            'weight': 'startingPitcherWeight'
        },
        inplace=True
    )
    
    # Write out
    outpath = base_dest + "{}/".format(date_url.replace("/", ""))
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    probable_starters.to_csv(
        outpath + "probableStarters.csv",
        index=False
    )
    probable_starters.to_parquet(
        outpath + 'probableStarters.parquet'
    )


if __name__ == "__main__":

    # COnfiguration

    # Run Log
    date = dt.datetime(year=2019, month=7, day=1)

    # Teams
    base_url = "http://gd2.mlb.com/components/game/mlb/"
    base_dest = "/Volumes/Transcend/00_gameday/"
    ref_dest = "/Volumes/Transcend/99_reference/"

    # Misc
    registry_hist = 10

    # Iterate over today and tomorrow
    dates = [date]
    for dd in dates:
        scrape_game_previews(dd)
