import os
import shutil
import pandas as pd


def add_game_id_to_parquet(path):
    """

    :param paths:
    :return:
    """

    # Iterate over files in each dir
    for fname in [x for x in os.listdir(path) if ".parquet" in x]:
        try:
            df = pd.read_parquet(path + "/" + fname)
            df.loc[:, 'gameId'] = df['game_id'].copy(deep=True)
            df = df.loc[:, ~df.columns.duplicated()]
            df.to_parquet(path + "/" + fname)
        except:
            pass


def add_game_id_to_csv(path):
    """
    
    """

    # Iterate over files in each dir
    for fname in [x for x in os.listdir(path) if ".csv" in x]:
        print(fname)
        df = pd.read_csv(path+"/"+fname, dtype=str)
        #df['game_id'] = str(path.split("/")[len(path.split("/"))-1])
        #df.to_csv(path+"/"+fname)
        df.to_parquet(path+"/"+fname.replace(".csv", ".parquet"))


if __name__ == "__main__":

    # Base path
    path = "/Volumes/Samsung_T5/mlb/gdApi/00_gameday/"

    # Get list of dirs containing files
    dirs = os.listdir(path)
    dirs = [x for x in dirs if all(y in x for y in ['year_2017', 'month_04', 'day_01'])]
    paths = [path + dir_ for dir_ in dirs]

    # Iterate over files in each dir
    for path in paths:
        add_game_id_to_parquet(path=path)
        #add_game_id_to_csv(path=path)
        
