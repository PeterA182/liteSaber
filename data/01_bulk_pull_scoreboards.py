import os
import sys
import numpy as np
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ETREE
import datetime as dt
import utilities as util
CONFIG = util.load_config()
pd.set_option('display.max_columns', 500)


def extract_game_scores(data):
    """
    """

    resp = data.getroot()

    # GameId
    score = 


def scrape_game_scoreboards(date):
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

    #
    # SCOREBOARD
    #
    scoreboards = []
    for gid in game_links:
        print("        {}".format(str(gid)))
        try:
            game_id = str(gid.get('href'))[7:]
            rbs = full_url + "/" + str(gid.get('href'))[7:] + "miniscoreboard.xml"
            resp = urllib.request.urlopen(rbs)
            resp = ETREE.parse(resp)
            df = extract_game_score(resp)
            df['gameId'] = game_id
            scoreboards.append(df)
        except ValueError as VE:
            score = pd.DataFrame()
            pass
    try:
        scoreboards = pd.concat(
            objs=scoreboards,
            axis=0
        )
    except ValueError as VE:
        scoreboards = pd.DataFrame()
        pass

    scoreboards.to_csv(
        base_dest + "{}/scoreboards.csv".format(date_url.replace("/", "")),
        index=False
    )
    scoreboards.to_parquet(
        base_dest +
        '{}/scoreboards.parquet'.format(date_url.replace("/", ""))
    )

if __name__ == "__main__":

    # COnfiguration
