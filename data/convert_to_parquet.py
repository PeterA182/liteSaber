import os
import shutil
import pandas as pd


def convert_csv_to_parquet(paths, remove_previous):
    """

    :param paths:
    :return:
    """

    # Iterate over files in each dir
    for path in paths:
        for fname in [x for x in os.listdir(path) if ".csv" in x]:
            print("Converting to parquet: {}".format(path+"/"+fname))
            df = pd.read_csv(path + "/" + fname)
            df.to_parquet(path + "/" + fname.replace(".csv", ".parquet"))
            if remove_previous:
                os.remove(path + "/" + fname)


def move_folders(source, destination, file_keys, remove_previous=False):
    """

    :param source:
    :param destination:
    :param remove_previous:
    :return:
    """

    from_ = [
        source + "/" + x for x in os.listdir(source) if all(
            y in x for y in file_keys
        )
    ]

    for sub_dir in from_:
        shutil.move(sub_dir, destination)

        if remove_previous:
            os.remove(sub_dir)



if __name__ == "__main__":

    # Base path
    path = "/Volumes/Transcend/gameday/"

    # Get list of dirs containing files
    dirs = os.listdir(path)
    dirs = [x for x in dirs if all(y in x for y in ['year_', 'month_', 'day_'])]
    paths = [path + dir_ for dir_ in dirs]

    # Call conversion
    convert_csv_to_parquet(paths=paths, remove_previous=True)