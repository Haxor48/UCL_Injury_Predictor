from pybaseball import pybaseball as pb
import pandas as pd
from pandas import DataFrame
import numpy as np
from pathlib import Path
import datetime
import requests

pb.cache.enable()

#season_tots = pb.pitching_stats(2015, 2023, qual=1)
#season_tots.to_csv(Path('season_tots.csv'))

season_tots = pd.read_csv(Path('season_tots.csv'))

# Build UCL injury table
def build_ucl_injuries():
    tj = pd.read_csv(Path('tj.csv'))
    ucl_prp = pd.read_csv(Path('ucl_prp.csv'))
    ucl_internal = pd.read_csv(Path('ucl_internal.csv'))

    def to_date_list(column):
        return list(map(str, column.tolist()))
    
    def get_injury_year(date):
        print(date)
        if (date == np.nan or type(date) == float):
            return np.nan
        if date.month < 4:
            return date.year-1
        return date.year

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
    names += ucl_prp[ucl_prp.TJ_Surgery_Date != np.nan]['Player'].tolist()
    names += ucl_internal[ucl_internal.TJ_Surgery_Date != np.nan]['Player'].tolist()
    dates = to_date_list(tj['Date'])
    dates += to_date_list(ucl_prp[ucl_prp.TJ_Surgery_Date != np.nan]['Date'])
    dates += to_date_list(ucl_internal[ucl_internal.TJ_Surgery_Date != np.nan]['Date'])
    dates = list_in_date(dates)
    
    years = []
    for date in dates:
        years.append(get_injury_year(date))

    data = {'Name': names,
            'Date':dates,
            'Year': years}

    ucl_injuries = DataFrame(data)

    ucl_injuries = ucl_injuries.drop_duplicates(subset=['Name', 'Date'], keep=False)
    ucl_injuries = ucl_injuries.dropna()
    ucl_injuries = ucl_injuries.sort_values(by='Date', ascending=False)

    ucl_injuries.to_csv(Path('ucl_injuries.csv'))

# Add injury to season totals
def add_ucl_injuries_to_table():
    ucl_injuries = pd.read_csv(Path('ucl_injuries.csv'))
    ucl_injury_season = []
    for index, row in season_tots.iterrows():
        if len(ucl_injuries[(ucl_injuries.Name == row['Name']) & (ucl_injuries.Year == row['Season'])]) > 0:
            ucl_injury_season.append(1)
        else:
            ucl_injury_season.append(0)
    season_tots['UCL_Injury'] = ucl_injury_season
    season_tots.to_csv(Path('season_tots.csv'))
    
def get_game_codes():
    all_games = []
    for i in range(2015, 2024):
        response = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={i}-01-01&endDate={i}-12-31&gameType=R&fields=dates,date,games,gamePk')
        json = response.json()
        for date in json['dates']:
            for game in date['games']:
                all_games.append(game['gamePk'])
    return all_games
