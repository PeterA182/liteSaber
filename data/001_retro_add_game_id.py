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
        df = pd.read_parquet(path + "/" + fname)
        df['game_id'] = str(path.split("/")[len(path.split("/"))-1])
        df.to_parquet(path + "/" + fname)
        print("'game_id' added to file: {}".format(str(fname)))


def add_game_id_to_csv(path):
    """
    
    """

    # Iterate over files in each dir
    for fname in [x for x in os.listdir(path) if ".csv" in x]:
        df = pd.read_csv(path+"/"+fname)
        df['game_id'] = str(path.split("/")[len(path.split("/"))-1])
        df.to_csv(path+"/"+fname)
        df.to_parquet(path+"/"+fname.replace(".csv", ".parquet"))


if __name__ == "__main__":

    # Base path
    path = "/Volumes/Transcend/gameday/"

    # Get list of dirs containing files
    dirs = os.listdir(path)
    dirs = [x for x in dirs if all(y in x for y in ['year_', 'month_', 'day_'])]
    paths = [path + dir_ for dir_ in dirs]

    # Iterate over files in each dir
    for path in paths:
        add_game_id_to_parquet(path=path)
        add_game_id_to_csv(path=path)
        
