import os
import gc
import sys
import numpy as np
import pandas as pd
import datetime as dt
import utilities as util
CONFIG = util.load_config()

if __name__ == "__main__":

    # Full dataset path
    year_min = 2018
    year_max = 2018
    teams_list = ['ari']
    feature_fill_threshold = 0.95
    idx_cols = ['away_team_code', 'home_team_code',
                'date', 'prev_gameid_merge_key']


    # Iterate
    for year in [
        yr for yr in np.arange(year_min, year_max+1, 1)
    ]:

        # Get list of teams that played this year
        year_teams = os.listdir(
            CONFIG.get('paths').get('initial_featurespaces')
        )
        year_teams = [
            fname.split("_")[1] for fname in year_teams
        ]

        # Read in all teams
        year_stats = pd.concat(
            objs=[
                pd.read_parquet(
                    CONFIG.get('paths').get('initial_featurespaces') + fname
                ) for fname in os.listdir(CONFIG.get('paths').get('initial_featurespaces'))
            ],
            axis=0
        )

        # Iterate over teams and create their table for the year
        for team in year_teams:
            print("Now Processing: {}".format(team))

            # Read in team's table
            df_curr_team = pd.read_parquet(
                CONFIG.get('paths').get('initial_featurespaces') + \
                "{}_{}_initial_featurespace.parquet".format(
                    str(year), str(team)
                )
            )

            #
            #
            # ---------------------------------------
            # Assemble featurespaces of home and away

            # Subset Home and Away
            # Current Team is Home
            df_curr_team_home = df_curr_team.loc[
                df_curr_team['home_team_code'] == team, :]
            away_competitors = list(set(df_curr_team_home['away_team_code']))
            df_curr_team_home.drop(
                labels=idx_cols, axis=1, inplace=True
            )
            df_curr_team_home.rename(
                columns={col: '{}_home'.format(col) for col in df_curr_team_home.columns
                         if col != 'gameId'},
                inplace=True)
            # Current Team is away
            df_curr_team_away = df_curr_team.loc[
                df_curr_team['away_team_code'] == team, :]
            home_competitors = list(set(df_curr_team_away['home_team_code']))
            df_curr_team_away.drop(
                labels=idx_cols, axis=1, inplace=True
            )
            df_curr_team_away.rename(
                columns={col: '{}_away'.format(col) for col in df_curr_team_away.columns
                         if col != 'gameId'},
                inplace=True
            )

            # -----------------------
            # Handle Away Competitors (competitor is away team)
            try:
                away_competitors = pd.concat(
                    objs=[
                        pd.read_parquet(
                            CONFIG.get('paths').get('initial_featurespaces')+fname
                        ) for fname in os.listdir(
                            CONFIG.get('paths').get('initial_featurespaces')
                        ) if fname.split("_")[1] in away_competitors
                    ],
                    axis=0
                )
            except ValueError as VE:
                continue
            away_competitors = away_competitors.loc[
                away_competitors['away_team_code'].isin(away_competitors),
            :]
            away_competitors.drop(labels=idx_cols, axis=1, inplace=True)
            away_competitors.rename(
                columns={col: '{}_away'.format(col) for col in away_competitors.columns
                         if col != 'gameId'},
                inplace=True
            )

            away_competitors = \
                away_competitors.loc[:, [
                    x for x in away_competitors.columns if (
                        np.mean(away_competitors[x].notnull()) >
                        feature_fill_threshold
                    ) or (x == 'gameId')
                ]]

            # -----------------------
            # Handle Home Competitors (competitor is home team)
            try:
                home_competitors = pd.concat(
                    objs=[
                        pd.read_parquet(
                            CONFIG.get('paths').get('initial_featurespaces')+fname
                        ) for fname in os.listdir(
                            CONFIG.get('paths').get('initial_featurespaces')
                        ) if fname.split("_")[1] in home_competitors
                    ],
                    axis=0
                )
            except ValueError as VE:
                continue
            home_competitors = home_competitors.loc[
                home_competitors['home_team_code'].isin(home_competitors),
            :]
            home_competitors.drop(labels=idx_cols, axis=1, inplace=True)
            home_competitors.rename(
                columns={col: '{}_home'.format(col) for col in home_competitors.columns
                         if col != 'gameId'},
                inplace=True
            )
            home_competitors = \
                home_competitors.loc[:, [
                    x for x in home_competitors.columns if (
                        np.mean(home_competitors[x].notnull()) >
                        feature_fill_threshold
                    ) or (x == 'gameId')
                ]]

            # Stack and merge
            df_curr_team_home = pd.merge(
                df_curr_team_home, away_competitors,
                how='left',
                on=['gameId'],
                validate='1:1'
            )
            del away_competitors
            gc.collect()
            
            df_curr_team_away = pd.merge(
                df_curr_team_away, home_competitors,
                how='left', on=['gameId'], validate='1:1'
            )
            del home_competitors
            gc.collect()
            df_curr_team = pd.concat(
                objs=[df_curr_team_home, df_curr_team_away],
                axis=0
            )

            # ---------------------------------------
            # Assemble Team Stats

            # Assemble Home Team Stats

            # Assemble Away Team Stats
            
            # ---------------------------------------
            # Add final target flag
            targets = pd.concat(
                objs=[
                    pd.read_csv(
                        CONFIG.get('paths').get('raw')+
                        dd+"/game_linescore_summary.csv",
                        dtype=str
                    ) for dd in os.listdir(CONFIG.get('paths').get('raw'))
                    if str(year) in dd and "game_linescore_summary.csv" in
                    os.listdir(CONFIG.get('paths').get('raw')+dd+"/")
                ],
                axis=0
            )
            print(targets[['home_team_runs', 'away_team_runs']].head(20))
            print("----------"*3)
            #targets.loc[:, 'home_team_runs'] = \
            #    targets['home_team_runs'].str.replace("", "0")
            targets.loc[:, 'home_team_runs'] = \
                targets['home_team_runs'].astype(float)
            print(targets[['home_team_runs']].head(20))
            #targets.loc[:, 'away_team_runs'] = \
            #    targets['away_team_runs'].str.replace("", "0")
            targets.loc[:, 'away_team_runs'] = \
                targets['away_team_runs'].astype(float)
            targets.loc[:, 'home_win'] = (
                targets['home_team_runs'] >
                targets['away_team_runs']
            ).astype(int)
            df_curr_team = pd.merge(
                df_curr_team,
                targets,
                how='left',
                on=['gameId'],
                validate='1:1'
            )

            # ---------------------------------------
            # Re-Sort
            df_curr_team.sort_values(
                by=['gameId'],
                ascending=True,
                inplace=True
            )

            # ---------------------------------------
            # Save 031
            df_curr_team.to_parquet(
                CONFIG.get('paths').get('full_featurespaces') + \
                "{}_{}_finished_featurespace.parquet".format(
                    str(year), str(team)
                )
            )
