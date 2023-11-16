from pybaseball import pybaseball as pb
import pandas as pd
from pandas import DataFrame
import numpy as np
from pathlib import Path
import datetime

pb.cache.enable()

#season_tots = pb.pitching_stats(2015, 2023, qual=1)
#season_tots.to_csv(Path('season_tots.csv'))

season_tots = pd.read_csv(Path('season_tots.csv'))

tj = pd.read_csv(Path('tj.csv'))
ucl_prp = pd.read_csv(Path('ucl_prp.csv'))
ucl_internal = pd.read_csv(Path('ucl_internal.csv'))

def to_date_list(column):
    return list(map(str, column.tolist()))

def list_in_date(list):
    new_list = []
    for item in list:
        split = item.split('/')
        if (split != ['nan']):
            new_list.append(datetime.date(int(split[2]), int(split[0]), int(split[1])))
        else:
            new_list.append(np.nan)
    return new_list

names = tj['Player'].tolist()
names += ucl_prp['Player'].tolist()
names += ucl_internal['Player'].tolist()
dates = to_date_list(tj['Date'])
dates += to_date_list(ucl_prp['Date'])
dates += to_date_list(ucl_internal['Date'])
dates = list_in_date(dates)

data = {'Name': names,
        'Date':dates}

ucl_injuries = DataFrame(data)

ucl_injuries = ucl_injuries.drop_duplicates(subset=['Name', 'Date'], keep=False)
ucl_injuries = ucl_injuries.dropna()
ucl_injuries = ucl_injuries.sort_values(by='Date', ascending=False)