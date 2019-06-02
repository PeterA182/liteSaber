import os
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ETREE
import datetime as dt
pd.set_option('display.max_columns', 500)


def extract_probables(data):
    """
    Extracts probable home and away pitchers from atv_preview.xml
    """

    resp = resp.getroot()
    pitcher_prev_name = resp[0][0][0][2][0][1][0].text
    pitcher_prev_stat = resp[0][0][0][2][0][1][1].text
    pitcher_prev_side = resp[0][0][0][2][0][1][2].text
    df = pd.DataFrame({'gameId': [game_id],
                       'probableStarterName': [pitcher_prev_name],
                       'probableStarterStat': [pitcher_prev_stat],
                       'probableStarterSide': [pitcher_prev_side]})
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
    req = BeautifulSoup(test_rep)
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
            probable_starters.append(df)
        except: ValueError as VE:
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

    probable_starters.to_csv(
        base_dest + "{}/probableStarters.csv".format(date_url.replace("/", "")),
        index=False
    )
    probable_starters.to_parquet(
        base_dest+
        '{}/probableStarters.csv'.format(date_url.replace("/", ""))
    )
    

if __name__ == "__main__":

    # COnfiguration

    # Run Log
    date = dt.datetime.now()

    # Teams
    base_url = "http://gd2.mlb.com/components/game/mlb/"
    base_dest = "/Volumes/Transcend/gameday/"

    # Iterate over today and tomorrow
    dates = [date, date+dt.timedelta(days=1)]
    for dd in dates:
        print("Getting Probable Starters From: {}".format(str(dd)))
        scrape_game_preview(dd)
