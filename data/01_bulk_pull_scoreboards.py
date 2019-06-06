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


def extract_inning_lines(data):
    """
    """

    # innings
    inning_lines_game = []
    data = data['data']['game']
    for inning_dict in data['linescore']:
        df = pd.DataFrame({k: [v] for k, v in inning_dict.items()})
        inning_lines_game.append(df)
    
    inning_lines_game = pd.concat(inning_lines_game, axis=0)
    return inning_lines_game


def extract_home_runs(data):
    """
    """

    # Home Runs
    home_runs_game = []
    for player, infos in data['data']['game']['home_runs'].items():
        for info in infos:
            try:
                df = pd.DataFrame({
                    k: [v] for k, v in info.items()
                })
            except AttributeError as AE:
                df = pd.DataFrame()
                pass
            home_runs_game.append(df)

    home_runs_game = pd.concat(home_runs_game, axis=0)
    return home_runs_game


def extract_pitcher_summaries(data):
    """
    """

    # Pitcher Summaries
    pitcher_summaries_game = []
    for pitcher_ in [x for x in data['data']['game'].keys() if x[-7:] == 'pitcher']:
        curr = data['data']['game'][pitcher_]
        df = pd.DataFrame({
            k: [v] for k, v in curr.items()
        })
        pitcher_summaries_game.append(df)
    pitcher_summaries_game = pd.concat(pitcher_summaries_game, axis=0)
    return pitcher_summaries_game


def extract_final_summaries(data):
    """
    """

    # Final Summaries
    details = [
        x for x in data['data']['game'].keys() if (any(
            y in x for y in [
                'away_', 'home_',
                'is_no_hitter', 'is_perfect_game'
            ]
        ) and ('home_run' not in x))
    ]
    df = pd.DataFrame({
        k: [data['data']['game'][k]] for k in details
    })
    return df
    


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
    # Line Score, Home Runs, Pitcher Summaries
    inning_lines = []
    home_runs = []
    pitcher_summaries = []
    final_summaries = []
    for gid in game_links:
        print("        {}".format(str(gid)))

        # Inning Line Scores, Home Runs, Pitchers
        game_id = str(gid.get('href'))[7:]
        rbs = full_url + "/" + str(gid.get('href'))[7:] + "linescore.json"
        resp = urllib.request.urlopen(rbs)
        resp = pd.read_json(resp)

        # Line Scores
        try:
            df = extract_inning_lines(resp)
            df['gameId'] = game_id
            inning_lines.append(df)
        except:
            pass

        # Home Runs
        try:
            df = extract_home_runs(resp)
            df['gameId'] = game_id
            home_runs.append(df)
        except:
            pass

        # Pitcher Summaries
        try:
            df = extract_pitcher_summaries(resp)
            df['gameId'] = game_id
            pitcher_summaries.append(df)
        except:
            pass

        # Final Summary
        try:
            df = extract_final_summaries(resp)
            df['gameId'] = game_id
            final_summaries.append(df)
        except:
            pass
        
            
        
    try:
        inning_lines = pd.concat(inning_lines, axis=0)
        home_runs = pd.concat(home_runs, axis=0)
        pitcher_summaries = pd.concat(pitcher_summaries, axis=0)
        final_summaries = pd.concat(final_summaries, axis=0)
        
        # Innign Lines
        inning_lines.to_csv(
            base_dest + "{}/inning_lines.csv".format(date_url.replace("/", "")),
            index=False
        )
        inning_lines.to_parquet(
            base_dest +
            '{}/inning_lines.parquet'.format(date_url.replace("/", ""))
        )

        # Home Runs
        home_runs.to_csv(
            base_dest + "{}/home_runs.csv".format(date_url.replace("/", "")),
            index=False
        )
        home_runs.to_parquet(
            base_dest +
            '{}/home_runs.parquet'.format(date_url.replace("/", ""))
        )

        # Pitcher Summaries
        pitcher_summaries.to_csv(
            base_dest + "{}/pitcher_summaries.csv".format(date_url.replace("/", "")),
            index=False
        )
        pitcher_summaries.to_parquet(
            base_dest +
            '{}/pitcher_summaries.parquet'.format(date_url.replace("/", ""))
        )

        # Final Summaries
        final_summaries.to_csv(
            base_dest + "{}/game_linescore_summary.csv".format(date_url.replace("/", "")),
            index=False
        )
        final_summaries.to_parquet(
            base_dest +
            '{}/final_summaries.parquet'.format(date_url.replace("/", ""))
        )
    except:
        print("      no games on day")
    

if __name__ == "__main__":

    # COnfiguration
    min_date = dt.datetime(year=2018, month=8, day=1)
    max_date = dt.datetime(year=2018, month=9, day=1)

    # Teams
    base_url = "http://gd2.mlb.com/components/game/mlb/"
    base_dest = "/Volumes/Transcend/gameday/"

    # Iterate over years
    years = [y for y in np.arange(min_date.year, max_date.year+1, 1)]
    dates = [min_date+dt.timedelta(days=i)
             for i in range((max_date-min_date).days+1)]

    for dd in dates:
        print("Scraping Linescore Summaries from: {}".format(str(dd)))
        scrape_game_scoreboards(dd)
