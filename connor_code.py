from pybaseball import pybaseball as pb
import pandas as pd
from pandas import DataFrame
import numpy as np
from pathlib import Path

pb.cache.enable()

#season_tots = pb.pitching_stats(2015, 2023, qual=1)
#season_tots.to_csv(Path('season_tots.csv'))

season_tots = pd.read_csv(Path('season_tots.csv'))

tj = pd.read_csv(Path('tj.csv'))
ucl_prp = pd.read_csv(Path('ucl_prp.csv'))
ucl_internal = pd.read_csv(Path('ucl_internal.csv'))

names = tj['Player'].tolist()
names.append(ucl_prp['Player'].tolist())
names.append(ucl_internal['Player'].tolist())
dates = tj['Date'].tolist()
dates.append(ucl_prp['Date'].tolist())
dates.append(ucl_internal['Date'].tolist())

data = {'Name': names,
        'Date':dates}

ucl_injuries = DataFrame(data)

#ucl_injuries.drop_duplicates(subset=['Name', 'Date'], keep=False)
ucl_injuries.dropna()
ucl_injuries.sort_values(by='Date', ascending=False)

print(ucl_injuries.head())