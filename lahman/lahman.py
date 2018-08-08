__author__ = 'Peter Altamura'

import os
import pandas as pd
import warnings
import numpy as np


# Class
class Lahman(object):
    """
    Lahman objects contain dictionaries of tables downloaded from the Lahman
    Baseball statistics resource online at:
    http:\\www.seanlahman.com/baseball-archive/
    PARAMETERS
    ----------
    base_dir: str
        string of directory path to dir containing Lahman DB files
    min_year: int
        integer lower bound of year range for which Lahman DB files
        should be read in
    max_year: int
        integer upper bound of year range for which Lahman DB files
        should be read in
    ATTRIBUTES
    ----------
    name_map: dict
        dictionary of raw filenames to names to be used as ref keys
    min_year: int
        integer lower bound of year range for which Lahman DB files
        should be read in
    max_year: int
        integer upper bound of year range for which Lahman DB files
        should be read in
    range: bool
        determined if Lahman obj contain tuple dict keys due to multiple years
        of files being contained
    NOTES
    -----
    If a range of years is passed:
        to Lahman object, the last directory in
        the base_dir path should contain sub-directories by year, each of which,
        in turn contain that year's edition of Lahman DB files.
        tables will have a 4-digit year suffix
    max_year is inclusive
    max_year=2012 will read in 2012's Lahman DB files
    RETURN
    ------
    Lahman class
    """

    def __getitem__(self, item):

        # If tuple item is passed, ensure types and return
        if type(item) == tuple:
            try:
                return self.dict[item]
            except KeyError:
                yr = item[0]
                yr = str(yr)
                tbl = item[1]
                tbl = str(tbl)
                return self.dict[(yr, tbl)]

        # If str of table name returned, ensure year PARAM is fulfilled
        elif type(item) == str:
            try:
                return self.dict[item]
            except:
                raise Warning(
                    """Please either enter a tuple key:
                        (\"2013\", \"table_name\")
                    """
                )


    def __setitem__(self, key, value):
        self.dict[key] = value


    def __init__(self, base_dir, min_year=None, max_year=None):
        self.base_dir = base_dir
        self.name_map = {
            'AllstarFull': 'all_star_full',
            'Appearances': 'appearances',
            'AwardsManagers': 'awards_managers',
            'AwardsPlayers': 'awards_players',
            'AwardsShareManagers': 'awards_share_managers',
            'AwardsSharePlayers': 'awards_share_players',
            'Batting': 'batting',
            'BattingPost': 'batting_post',
            'CollegePlaying': 'college_playing',
            'Fielding': 'fielding',
            'FieldingOF': 'fielding_outfield',
            'FieldingPost': 'fielding_post',
            'HallOfFame': 'hall_of_fame',
            'Managers': 'managers',
            'ManagersHalf': 'managers_half',
            'Master': 'master',
            'Pitching': 'pitching',
            'PitchingPost': 'pitching_post',
            'Salaries': 'salaries',
            'Schools': 'schools',
            'SchoolsPlayers': 'schools_players',
            'SeriesPost': 'series_post',
            'Teams': 'teams',
            'TeamsFranchises': 'teams_franchises',
            'TeamsHalf': 'teams_half'
        }
        self.dict = {}
        self.min_year = min_year
        self.max_year = max_year
        self.range = False if (min_year == None and max_year == None) else True

        #   ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----   #
        #   PARAMETER CHECKS
        # base_dir format check
        if base_dir[-1:] != '\\':
            base_dir = base_dir + '\\'

        # Make sure of full range or no range
        if (min_year == None and max_year != None) or \
            (min_year!= None and max_year == None):
            raise Warning("If a range is to be given, a \'min_year\' and "
                          "\'max_year\' must both be given")

        #
        #   ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----   #
        #   Begin finding and updating self.dict with tables found in Lahman dir
        #   or in year sub-dirs within Lahman dir
        #

        # ---- ---- IF RANGE GIVEN ---- ---- #
        # Being in multiple years of Lahman DB in same dict
        if min_year != None or max_year != None:
            warnings.warn(
        """If years param is != None, the following requirements are applied:
            1) file_struct for Lahman files must be the following:
                Example file structure with CSVs:
                base_dir = Drive:\\...\\{name of lahman dir}\\{year}\\
                    ...\\lahman_files\\2009\\:
                        Managers.csv
                        Appearances.csv
                        ...
                        TeamsFranchises.csv
                        TeamsHalf.csv
                    ...\\lahman_files\\2010\\:
                        Managers.csv
                        Appearances.csv
                        ...
                        TeamsFranchises.csv
                        TeamsHalf.csv
                    and so forth...
        """)

            # Use year range to determine sub-directories to enter
            yr_range = np.arange(
                start=min_year,
                stop=max_year+1,
                step=1)

            # Iterate by year
            for yr in yr_range:

                # establish path to year sub_dir
                sub_path = base_dir + "{}\\".format(
                    str(yr)
                )

                # Look through and pull tables
                for file in os.listdir(sub_path):

                    # establish full path and make sure a CSV
                    if str(file[-4:]) == '.csv':

                        # Assemble full
                        full_path = sub_path + file

                        # est table
                        table = pd.read_csv(full_path)

                        # Append
                        self.dict[(str(yr), self.name_map[file[:-4]])] = table

        # If not reading in range of years' worth of files
        elif not self.range:

            # Iterate over files and populate attribute
            for file in os.listdir(base_dir):

                # establish full path and make sure a CSV
                if str(file[-4:]) == '.csv':

                    # Assemble full path to file
                    full_path = base_dir + file

                    # est table
                    table = pd.read_csv(full_path)

                    # Append
                    self.dict[self.name_map[file[:-4]]] = table


    @property
    def tables(self):
        return list(self.dict.keys())


    @property
    def years(self):
        if self.range:
            return set([x[0] for x in self.dict.keys()])


    def find_table(self, year=None, begins_with=None,
                   ends_with=None, contains=None):
        """
        Search through table names in the Lahman directory to find name of table
        to call
        PARAMETERS
        ----------
        year: str
            year to search for in strings of table names to be returned
        begins_with: str
            search for tables with name that begin with passed string
        ends_with: str
            search for tables with name that end with passed string
        contains: str
            search for tables with name that contain the passed string
        """

        # Return List
        matching_tbls = []

        if year == None:
            print(self.tables)

            for file in self.tables:

                # Begins with
                if begins_with != None:
                    print(str(file[1]))
                    if str(file[1][0:len(begins_with)]) == begins_with:
                        matching_tbls.extend([file])

                # Ends with
                if ends_with != None:
                    if str(file[1][0-len(ends_with):]) == ends_with:
                        matching_tbls.extend([file])

                # Contains
                if contains != None:
                    if str(contains) in str(file[1]):
                        matching_tbls.extend([file])

        # Year param
        if year != None:

            # Search CSV dir
            for file in self.tables:

                # Begins with
                if begins_with != None:
                    if str(file[0:len(begins_with)]) == begins_with:
                        matching_tbls.extend([file])

                # Ends with
                if ends_with != None:
                    if str(file[0-len(ends_with):]) == ends_with:
                        matching_tbls.extend([file])

                # Contains
                if contains != None:
                    if str(contains) in str(file):
                        matching_tbls.extend([file])

        # Ret assembled list
        return matching_tbls