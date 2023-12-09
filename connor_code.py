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
    for i in range(2022, 2024):
        response = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={i}-01-01&endDate={i}-12-31&gameType=R&fields=dates,date,games,gamePk')
        json = response.json()
        for date in json['dates']:
            for game in date['games']:
                all_games.append(game['gamePk'])
    return all_games
    
def update_game_by_game():

    def convert_to_dataframe(name, pitch):
        output = {'pitch_name': name, **pitch}
        for field in output:
            output[field] = [output[field]]
        #print(DataFrame(output.update(pitch)))
        return DataFrame.from_dict(output)

    codes = get_game_codes()
    fields = ['release_speed', 'release_pos_x', 'release_pos_y', 'spin_dir', 'spin_rate_deprecated', 
            'break_angle_deprecated', 'break_length_deprecated']
    pitcher_logs = {}
    for code in codes[:1]:
        game = pb.statcast_single_game(code)
        if game is None:
            continue
        pitchers = {}
        for index, row in game.iterrows():
            print(pb.playerid_reverse_lookup([row['pitcher']], key_type='mlbam'))
            if row['pitcher'] in pitchers:
                if row['pitch_name'] in pitchers[row['pitcher']]:
                    for field in fields:
                        pitchers[row['pitcher']][row['pitch_name']][field] += row[field]
                    pitchers[row['pitcher']][row['pitch_name']]['pitch_num'] += 1
                else:
                    pitchers[row['pitcher']][row['pitch_name']] = {'pitch_num': 1}
                    pitchers[row['pitcher']][row['pitch_name']]['pitch_type'] = row['pitch_type']
                    pitchers[row['pitcher']][row['pitch_name']]['game_date'] = row['game_date']
                    pitchers[row['pitcher']][row['pitch_name']] = {**pitchers[row['pitcher']][row['pitch_name']], **{field: row[field] for field in fields}}
            else:
                pitchers[row['pitcher']] = {row['pitch_name']: {'pitch_num': 1}}
                pitchers[row['pitcher']][row['pitch_name']]['pitch_type'] = row['pitch_type']
                pitchers[row['pitcher']][row['pitch_name']]['game_date'] = row['game_date']
                pitchers[row['pitcher']][row['pitch_name']] = {**pitchers[row['pitcher']][row['pitch_name']], **{field: row[field] for field in fields}}

        for pitcher in pitchers:
            for pitch in pitchers[pitcher]:
                for field in fields:
                    pitchers[pitcher][pitch][field] /= pitchers[pitcher][pitch]['pitch_num']
                    
        for pitcher in pitchers:
            if pitcher not in pitcher_logs:
                pitcher_logs[pitcher] = None
                for pitch in pitchers[pitcher]:
                    if pitcher_logs[pitcher] is None:
                        pitcher_logs[pitcher] = convert_to_dataframe(pitch, pitchers[pitcher][pitch])
                    else:
                        pitcher_logs[pitcher] = pd.concat([pitcher_logs[pitcher], convert_to_dataframe(pitch, pitchers[pitcher][pitch])])
            else:
                for pitch in pitchers[pitcher]:
                    pitcher_logs[pitcher]= pd.concat([pitcher_logs[pitcher], convert_to_dataframe(pitch, pitchers[pitcher][pitch])])
                    
    ##for pitcher in pitcher_logs:
        ##pitcher_logs[pitcher].to_csv(Path(f'.\\pitcher_logs\\{pitcher}.csv'))

def remove_accents(name):
        temp = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n').replace('ü', 'u').replace('Á', 'A')
        return temp.replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U').replace('Ñ', 'N').replace('Ü', 'U')
        
output = pb.playerid_reverse_lookup([660271], key_type='mlbam')
name = remove_accents(output.iloc[0]['name_first']) + ' ' + remove_accents(output.iloc[0]['name_last'])
print(name)