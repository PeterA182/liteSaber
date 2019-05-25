__author__ = 'Peter Altamura'

import os
import sys
sys.path.append("E:\\liteSaberPackage\\")
import urllib2
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ETREE
import numpy as np
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', 500)


def unpack_innings(game_tree):
    """

    :param game_tree:
    :return:
    """
    finished = []
    game_root = game_tree.getroot()
    game_dict = {"game_{}".format(k): v for k, v in game_root.attrib.items()}
    for inning_full in game_root:
        inning_full_dict = {'inning_{}'.format(k): v for k, v
                            in inning_full.attrib.items()}
        for inning_half in inning_full:
            side = inning_half.tag

            for atbat in inning_half:
                atbat_dict = {'atbat_{}'.format(k): v for k, v in atbat.attrib.items()}
                if atbat.tag == 'action':
                    continue

                try:
                    pitches = pd.concat(
                        objs=[
                            pd.DataFrame({"pitch_{}".format(k): [v] for k, v in
                                          pitch.attrib.items()}) for pitch in
                            atbat if pitch.tag == 'pitch'
                        ],
                        axis=0
                    )
                    for k, v in atbat_dict.items():
                        pitches[k] = v
                    for k, v in inning_full_dict.items():
                        pitches[k] = v
                    for k, v in game_dict.items():
                        pitches[k] = v
                    finished.append(pitches)
                except ValueError as VE:
                    pitches = pd.DataFrame()
                    pass
                try:
                    runners = pd.concat(
                        objs=[
                            pd.DataFrame({k: [v] for k, v in pitch.attrib.items()})
                            for pitch in atbat if pitch.tag == 'runner'
                        ],
                        axis=0
                    )
                    for k, v in atbat_dict.items():
                        runners[k] = v
                    for k, v in inning_full_dict.items():
                        runners[k] = v
                    for k, v in game_dict.items():
                        runners[k] = v
                    finished.append(runners)
                except ValueError as VE:
                    runners = pd.DataFrame()
                    pass
    finished = pd.concat(objs=finished, axis=0)
    return finished


def unpack_events(table):
    """

    :param table:
    :return:
    """

    return 0


def scrape_game_date(date):
    """

    :param date:
    :return:
    """
    ignore_keys = ['inning_line_score', 'batting', 'pitching', 'linescore']
    date_url = "year_{}/month_{}/day_{}".format(
        str(date.year).zfill(4),
        str(date.month).zfill(2),
        str(date.day).zfill(2)
    )
    full_url = base_url + date_url
    print(full_url)
    
    # Generate list of gid links
    test_resp = urllib2.urlopen(full_url)
    req = BeautifulSoup(test_resp)
    game_links = [x for x in req.find_all('a') if
                  str(x.get('href'))[7:10] == 'gid']

    #
    # BOXSCORE
    #
    date_games = []
    batting = []
    pitching = []
    innings = []
    for gid in game_links:
        print("        {}".format(str(gid)))
        try:
            game_id = str(gid.get('href'))[7:]
            rbs = full_url + "/" + str(gid.get('href'))[7:] + "boxscore.json"
            data_master = urllib2.urlopen(rbs)
            data_master = pd.read_json(data_master)
            data_master = data_master['data'].iloc[0]

            # Boxscore (data_master is a dictionary)
            df_box = pd.DataFrame({'gameId': [game_id]})
            df_box = pd.concat(
                objs=[df_box,
                      pd.DataFrame({k: [''.join([i if ord(i) < 128 else ' ' for i in v])]
                                    for k, v in data_master.items() if k not in ignore_keys})],
                axis=1
            )
            df_box['game_id'] = game_id
            date_games.extend([df_box])
            
            # Batting Details
            df_bat = data_master['batting']
            for team in df_bat:
                team_batting = pd.DataFrame({
                    "team_{}".format(k): [v for i in range(len(team['batter']))]
                    for k, v in team.items() if k not in [
                        'batter', 'text_data', 'text_data_es',
                        'note', 'note_es'
                    ]
                })
                batters = pd.concat(
                    objs=[pd.DataFrame({k: [''.join([i if ord(i) < 128 else ' ' for i in v])]
                                        for k, v in batter.items()})
                          for batter in team['batter']],
                    axis=0,
                    ignore_index=True
                )
                team_batting = pd.concat(
                    objs=[team_batting, batters],
                    axis=1
                )
                team_batting['game_id'] = game_id
                batting.append(team_batting)
            
            # Pitching Details
            df_ptch = data_master['pitching']
            for team in df_ptch:
                team_pitching = pd.DataFrame({
                    "team_{}".format(k): [v for i in range(len(team['pitcher']))]
                    for k, v in team.items() if k not in [
                    'pitcher', 'text_data', 'text_data_es',
                    'note', 'note_es'
                ]
                })
                pitchers = pd.concat(
                    objs=[pd.DataFrame(
                        {k: [''.join([i if ord(i) < 128 else ' ' for i in v])]
                         for k, v in batter.items()})
                          for batter in team['pitcher']],
                    axis=0,
                    ignore_index=True
                )
                team_pitching = pd.concat(
                    objs=[team_pitching, pitchers],
                    axis=1
                )
                team_pitching['game_id'] = game_id
                pitching.append(team_pitching)

            # Inning Details
            rbs = full_url + "/" + str(gid.get('href'))[7:] + "inning/inning_all.xml"
            innings_ret = unpack_innings(ETREE.parse(urllib2.urlopen(rbs)))
            innings.append(innings_ret)

        except Exception as E:
            print(E)
            pass

    if not os.path.exists(base_dest + '{}/'.format(date_url.replace("/", ""))):
        os.makedirs(base_dest + '{}/'.format(date_url.replace("/", "")))

    with open(base_dest + '{}/log_file.txt'.format(date_url.replace("/", "")), 'w+') as log:
        log.write("\n".join(str(L) for L in game_links))
        log.close()
    try:
        date_games = pd.concat(date_games, axis=0)
        date_games.to_csv(base_dest + '{}/boxscore.csv'.format(date_url.replace("/", "")),
                          index=False)
        batting = pd.concat(batting, axis=0)
        batting.to_csv(base_dest + '{}/batting.csv'.format(date_url.replace("/", "")),
                       index=False)
        pitching = pd.concat(pitching, axis=0)
        pitching.to_csv(base_dest + '{}/pitching.csv'.format(date_url.replace("/", "")),
                        index=False)
        innings = pd.concat(innings, axis=0)
        innings.to_csv(base_dest + '{}/innings.csv'.format(date_url.replace("/", "")),
                       index=False)
    except ValueError as VE:
        print("     no games on day")

    #
    # GAME_EVENTS
    #


if __name__ == "__main__":

    # Configuration
    #CONFIG = parse_config("./configuration.json")

    # Run Log
    min_date = dt.datetime(year=2019, month=2, day=1)
    max_date = dt.datetime(year=2019, month=5, day=22)

    # Teams
    teams = []
    base_url = "http://gd2.mlb.com/components/game/mlb/"
    base_dest = "/Volumes/Transcend/gameday/"

    # Iterate over years
    years = [y for y in np.arange(min_date.year, max_date.year+1, 1)]
    dates = [min_date+dt.timedelta(days=i)
             for i in range((max_date-min_date).days+1)]

    for dd in dates:
        print("Scraping games from: {}".format(str(dd)))
        scrape_game_date(dd)




