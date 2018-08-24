import os
import sys

import pandas as pd
import datetime as dt


def get_gameday_subdirs(path, window_start, window_end,
                        left_incl=True, right_incl=True):
    """

    :param window_start:
    :param window_end:
    :param left_incl:
    :param right_incl:
    :return:
    """

    # Gamedays

    gamedays = [
        x for x in os.listdir(path) if (
            (dt.datetime(year=int(x[5:9]),
                         month=int(x[15:17]),
                         day=int(x[-2:])) >=
             window_start)
            &
            (dt.datetime(year=int(x[5:9]),
                         month=int(x[15:17]),
                         day=int(x[-2:])) <=
             window_end)
        )
    ]

    return gamedays
