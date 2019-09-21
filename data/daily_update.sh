#!/bin/bash
cd /Users/peteraltamura/Documents/GitHub/liteSaber/data/
python 000_bulk_pull.py
python 010_standardize.py
python 02_batter_stats_new.py
python 02_pitcher_stats_new.py

