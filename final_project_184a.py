import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, mean_squared_error, log_loss, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from pybaseball import pybaseball as pb
import pandas as pd
from pandas import DataFrame
import numpy as np
from pathlib import Path
import datetime
import requests
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, SimpleRNN, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

pb.cache.enable()

def get_injury_year(date):
  if (date == np.nan or type(date) == float):
    return np.nan
  if date.month < 4:
    return date.year-1
  return date.year

def build_ucl_injuries():
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

    return ucl_injuries

# Add injury to season totals
def add_ucl_injuries_to_table(ucl_injuries):
    season_tots = pb.pitching_stats(2018, 2023, qual=1)
    ucl_injury_season = []
    for index, row in season_tots.iterrows():
        if len(ucl_injuries[(ucl_injuries.Name == row['Name']) & (ucl_injuries.Year == row['Season'])]) > 0:
            ucl_injury_season.append(1)
        else:
            ucl_injury_season.append(0)
    season_tots['UCL_Injury'] = ucl_injury_season
    season_tots.to_csv(Path('season_tots.csv'))
    return season_tots

import_cols_left = ['Name', 'Season', 'Age', 'G', 'GS', 'CG', 'IP', 'Pitches', 'K/BB', 'FB%', 'FBv',
                    'SL%', 'SLv', 'CT%', 'CTv', 'CB%', 'CBv', 'CH%', 'CHv', 'SF%', 'SFv', 'KN%', 'KNv',
                    'Zone%', 'F-Strike%', 'K%', 'BB%', 'UCL_Injury']
import_cols_right = ['last_name, first_name', 'Year', 'ff_avg_spin', 'si_avg_spin', 'fc_avg_spin', 'sl_avg_spin',
                     'ch_avg_spin', 'cu_avg_spin']

def remove_accents(name):
        temp = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n').replace('ü', 'u').replace('Á', 'A')
        return temp.replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U').replace('Ñ', 'N').replace('Ü', 'U')

def fix_name(name):
  split = name.split(', ')
  return remove_accents(split[1]) + ' ' + remove_accents(split[0])

def get_spin():
  output = None
  for i in range(2018, 2024):
    statcast = pb.statcast_pitcher_pitch_arsenal(i, 1, 'avg_spin')
    statcast['Year'] = [i] * len(statcast)
    for i in range(len(statcast)):
      statcast.at[i, 'last_name, first_name'] = fix_name(statcast.at[i,'last_name, first_name'])
    if output is None:
      output = statcast
    else:
      output = pd.concat([output, statcast])
  return output

def merge_sections(season, spin):
  season_locs = []
  spin_locs = []
  for col in import_cols_left:
    season_locs.append(season.columns.get_loc(col))
  for col in import_cols_right:
    spin_locs.append(spin.columns.get_loc(col))
  season = season.drop(season.columns[[x for x in range(len(season.columns)) if x not in season_locs]], axis=1)
  spin = spin.drop(spin.columns[[x for x in range(len(spin.columns)) if x not in spin_locs]], axis=1)
  merged = pd.merge(season, spin, how='outer', left_on = ['Name', 'Season'], right_on = ['last_name, first_name', 'Year'])
  merged = merged.drop(merged.columns[[merged.columns.get_loc('Season'), merged.columns.get_loc('Name'),
                                       merged.columns.get_loc('Year'), merged.columns.get_loc('last_name, first_name')]], axis=1)
  return merged

def aggregate_merged(merged):
  maxVelo = []
  medAvrVelo = []
  slowVelo = []
  avrSpin = []
  med_vars = ['SLv', 'CTv', 'SFv']
  slow_vars = ['CBv', 'CHv', 'KNv']
  spin_vars = ['ff_avg_spin', 'si_avg_spin', 'fc_avg_spin', 'sl_avg_spin',
                     'ch_avg_spin', 'cu_avg_spin']
  for index, row in merged.iterrows():
    max = row['FBv']
    tot_med = 0
    count_med = 0
    for var in med_vars:
      if ~np.isnan(row[var]):
        tot_med += row[var]
        count_med += 1
        if np.isnan(max) or max < row[var]:
          max = row[var]
    tot_slow = 0
    count_slow = 0
    for var in slow_vars:
      if ~np.isnan(row[var]):
        tot_slow += row[var]
        count_slow += 1
    tot_spin = 0
    count_spin = 0
    for var in spin_vars:
      if ~np.isnan(row[var]):
        tot_spin += row[var]
        count_spin += 1
    maxVelo.append(max)
    if count_med != 0:
      medAvrVelo.append(tot_med/count_med)
    else:
      medAvrVelo.append(np.nan)
    if count_slow != 0:
      slowVelo.append(tot_slow/count_slow)
    else:
      slowVelo.append(np.nan)
    if count_spin != 0:
      avrSpin.append(tot_spin/count_spin)
    else:
      avrSpin.append(np.nan)
  merged['MaxVelo'] = maxVelo
  merged['AvgMedVelo'] = medAvrVelo
  merged['AvgSlowVelo'] = slowVelo
  merged['AvgSpin'] = avrSpin
  to_drop = ['FBv','SL%', 'SLv', 'CT%', 'CTv', 'CB%', 'CBv', 'CH%', 'CHv',
             'SF%', 'SFv', 'KN%', 'KNv', 'ff_avg_spin', 'si_avg_spin',
             'fc_avg_spin', 'sl_avg_spin', 'ch_avg_spin', 'cu_avg_spin']
  for drop in to_drop:
    merged = merged.drop(merged.columns[[merged.columns.get_loc(drop)]], axis=1)
  return merged

def build_and_clean_long_term():
    ucl_injuries = build_ucl_injuries()
    season_tots = add_ucl_injuries_to_table(ucl_injuries)
    spin = get_spin()
    merged = merge_sections(season_tots, spin)
    merged = aggregate_merged(merged)
    merged_no_nan = merged.dropna()
    return merged, merged_no_nan

def get_X_Y(merged, merged_no_nan):
    return merged_no_nan.drop(merged.columns[[merged.columns.get_loc('UCL_Injury')]], axis=1), merged_no_nan['UCL_Injury']

def split_train_test(X, Y):
    return train_test_split(X, Y, test_size = 0.2, train_size = 0.8)

def get_best_k(X_train, Y_train, X_test, Y_test):

  K = [1, 2, 3, 4, 5, 6, 7, 8]

  errTrain = [0] * 8
  errTest = [0] * 8

  for i, k in enumerate(K):
    neighbors = KNeighborsClassifier(n_neighbors = k)
    model = neighbors.fit(X_train, Y_train)
    Y_test_pred = neighbors.predict(X_test)
    Y_train_pred = neighbors.predict(X_train)
    errTrain[i] = log_loss(Y_train, Y_train_pred)
    errTest[i] = log_loss(Y_test, Y_test_pred)
  plt.plot(K, errTrain, K, errTest)

def knn(X_train, Y_train, X_test, Y_test):
    knn = KNeighborsClassifier(n_neighbors=3)

    cv_scores = cross_val_score(knn, X_train, Y_train, cv=5)

    knn.fit(X_train.values, Y_train.values)

    Y_test_pred = knn.predict(X_test.values)

    accuracy = accuracy_score(Y_test.values, Y_test_pred)
    report = classification_report(Y_test, Y_test_pred)

    print("Test Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("Cross-validation Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())
    
def logistic_regression(X_train, Y_train, X_test, Y_test):
    model = LogisticRegression()

    cv_scores = cross_val_score(model, X_train, Y_train, cv=5)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)

    print("Test Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("Cross-validation Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())
    
def mlp(X_train, Y_train, X_test, Y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=300)

    cv_scores = cross_val_score(mlp, X_train, Y_train, cv=5)

    mlp.fit(X_train, Y_train)

    Y_pred = mlp.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)

    print("MLP Classifier Test Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("Cross-validation Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())
    
def cnn_with_validation(X, Y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Y_categorical = to_categorical(Y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = np.expand_dims(X_scaled[train_index], -1), np.expand_dims(X_scaled[test_index], -1)
        Y_train, Y_test = Y_categorical[train_index], Y_categorical[test_index]

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(Y_train.shape[1], activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test), verbose=0)

        scores = model.evaluate(X_test, Y_test, verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}')
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])

        fold_no += 1
        
    print('\nAverage scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    
def get_game_codes():
    all_games = []
    for i in range(2022, 2024):
        response = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={i}-01-01&endDate={i}-12-31&gameType=R&fields=dates,date,games,gamePk')
        json = response.json()
        for date in json['dates']:
            for game in date['games']:
                all_games.append(game['gamePk'])
    return all_games

fields = ['release_speed', 'release_pos_x', 'release_pos_y']

def injured_year(ucl_injuries, player, year):
      output = pb.playerid_reverse_lookup([player], key_type='mlbam')
      name = remove_accents(output.iloc[0]['name_first']) + ' ' + remove_accents(output.iloc[0]['name_last'])
      for index, row in ucl_injuries[ucl_injuries.Year == year].iterrows():
        if row['Name'].lower() == name:
          return 1
      return 0

def update_game_by_game(ucl_injuries):
    
    pitcher_logs = {}

    def convert_to_dataframe(name, pitch):
        output = {'pitch_name': name, **pitch}
        for field in output:
            output[field] = [output[field]]
        #print(DataFrame(output.update(pitch)))
        return DataFrame.from_dict(output)

    codes = get_game_codes()
    fields = ['release_speed', 'release_pos_x', 'release_pos_y']
    for code in codes:
        game = None
        try:
          game = pb.statcast_single_game(code)
        except:
          continue
        if game is None:
            continue
        pitchers = {}
        for index, row in game.iterrows():
            if row['pitcher'] in pitchers:
                if row['pitch_name'] in pitchers[row['pitcher']]:
                    for field in fields:
                        pitchers[row['pitcher']][row['pitch_name']][field] += row[field]
                    pitchers[row['pitcher']][row['pitch_name']]['pitch_num'] += 1
                else:
                    pitchers[row['pitcher']][row['pitch_name']] = {'pitch_num': 1}
                    pitchers[row['pitcher']][row['pitch_name']]['pitch_type'] = row['pitch_type']
                    pitchers[row['pitcher']][row['pitch_name']]['game_date'] = row['game_date']
                    pitchers[row['pitcher']][row['pitch_name']]['injured'] = pitchers[row['pitcher']][list(pitchers[row['pitcher']].keys())[0]]['injured']
                    pitchers[row['pitcher']][row['pitch_name']] = {**pitchers[row['pitcher']][row['pitch_name']], **{field: row[field] for field in fields}}
            else:
                pitchers[row['pitcher']] = {row['pitch_name']: {'pitch_num': 1}}
                pitchers[row['pitcher']][row['pitch_name']]['pitch_type'] = row['pitch_type']
                pitchers[row['pitcher']][row['pitch_name']]['game_date'] = row['game_date']
                pitchers[row['pitcher']][row['pitch_name']]['injured'] = injured_year(ucl_injuries, row['pitcher'], get_injury_year(row['game_date']))
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
    return pitcher_logs
                    
def save_pitchers(pitcher_logs):
  with open('pitcher_data/pitcher_list.txt', 'w') as w:
    for pitcher in pitcher_logs:
      pitcher_logs[pitcher].to_csv(f'pitcher_data/{pitcher}.csv', index=False)
      w.write(str(pitcher) + '\n')

def fix_injuries(ucl_injuries, pitcher_logs):
  for pitcher in pitcher_logs:
    count = 0
    for index, row in pitcher_logs[pitcher].iterrows():
      pitcher_logs[pitcher].at[count, 'injured'] = injured_year(ucl_injuries, pitcher, get_injury_year(pd.to_datetime(row['game_date'], format='%Y-%m-%d')))
      count += 1
      
def load_pitchers():
    pitcher_logs = {}
    with open('pitcher_data/pitcher_list.txt', 'r') as r:
        for pitcher in r:
            pitcher_logs[int(pitcher[:-1])] = pd.read_csv(Path(f'pitcher_data/{pitcher[:-1]}.csv'))
    return pitcher_logs

def load_delta():
    pitcher_logs = {}
    with open('pitcher_data/pitcher_list.txt', 'r') as r:
        for pitcher in r:
            pitcher_logs[int(pitcher[:-1])] = pd.read_csv(Path(f'pitcher_delta/{pitcher[:-1]}.csv'))
    return pitcher_logs

def first_occurence(dataframe, pitch, date):
  counter = 0
  for index, row in dataframe.iterrows():
    if row['pitch_name'] == pitch and row['game_date'] != date:
      return counter
    counter += 1
  return -1

def calc_difference(dataframe, row, firstIndex):
  tot_fields = ['pitch_num'] + fields
  if firstIndex == -1:
    return DataFrame({'pitch_name': [row['pitch_name']], 'injured': [row['injured']], 'game_date': [0],
                      **{field: [0] for field in tot_fields}})
  return DataFrame({'pitch_name': [row['pitch_name']], 'injured': [row['injured']],
                    'game_date': [(pd.to_datetime(row['game_date'], format='%Y-%m-%d')-pd.to_datetime(dataframe.iloc[firstIndex]['game_date'], format='%Y-%m-%d')).days],
                    **{field: [float(row[field]) - float(dataframe.iloc[firstIndex][field])] for field in tot_fields}})

def update_to_delta(pitcher_logs):
    for pitcher in pitcher_logs:
        new = None
        reversed = pitcher_logs[pitcher].iloc[::-1]
        while len(reversed) > 0:
            temp = reversed.iloc[0]
            reversed = reversed.iloc[1:, :]
            if new is None:
                new = calc_difference(reversed, temp, first_occurence(reversed, temp['pitch_name'], temp['game_date']))
            else:
                new = pd.concat([new, calc_difference(reversed, temp, first_occurence(reversed, temp['pitch_name'], temp['game_date']))])
        pitcher_logs[pitcher] = new[::-1]
    return pitcher_logs


def save_delta(pitcher_logs):
  for pitcher in pitcher_logs:
    if type(pitcher) == str:
      pitcher_logs[int(pitcher[:-1])].to_csv(f'pitcher_delta/{int(pitcher[:-1])}.csv', index=False)
    else:
      pitcher_logs[pitcher].to_csv(f'pitcher_delta/{pitcher}.csv', index=False)

def drop_na(pitcher_logs):
    to_drop = ['spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated']
    for pitcher in pitcher_logs:
        for drop in to_drop:
            pitcher_logs[pitcher] = pitcher_logs[pitcher].drop(pitcher_logs[pitcher].columns[[pitcher_logs[pitcher].columns.get_loc(drop)]], axis=1)
    return pitcher_logs

def rnn_with_validation(pitcher_logs):
    all_pitcher_data = pd.concat(pitcher_logs.values())

    features = all_pitcher_data.drop(['injured', 'pitch_name', 'pitch_type', 'game_date'], axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    target = all_pitcher_data['injured']
    target = to_categorical(target)

    num_samples = scaled_features.shape[0]
    num_timesteps = 1
    num_features = scaled_features.shape[1]
    scaled_features = scaled_features.reshape((num_samples, num_timesteps, num_features))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []

    for train_index, test_index in kf.split(scaled_features):
        X_train, X_test = scaled_features[train_index], scaled_features[test_index]
        Y_train, Y_test = target[train_index], target[test_index]

        model = Sequential()
        model.add(SimpleRNN(50, input_shape=(num_timesteps, num_features), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(50, return_sequences=False))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(target.shape[1], activation='softmax')) 

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test), verbose=0)

        scores = model.evaluate(X_test, Y_test, verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}')
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])

        fold_no += 10

    print('\nAverage scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    
########################################################################################    
##Call theses functions when doing general testing    

def long_term_study():
    merged, merged_no_nan = build_and_clean_long_term()
    X, Y = get_X_Y(merged, merged_no_nan)
    knn(split_train_test(X, Y))
    logistic_regression(split_train_test(X, Y))
    mlp(split_train_test(X, Y))
    cnn_with_validation(X, Y)

def short_term_study_files():
    pitcher_logs = load_delta()
    pitcher_logs = drop_na(pitcher_logs)
    rnn_with_validation(pitcher_logs)
    
def short_term_stufy_api():
    #Not recommended, will take at least 3 hours to run
    ucl_injuries = build_ucl_injuries()
    pitcher_logs = update_game_by_game(ucl_injuries)
    pitcher_logs = fix_injuries(ucl_injuries, pitcher_logs)
    pitcher_logs = update_to_delta(pitcher_logs)
    pitcher_logs = drop_na(pitcher_logs)
    rnn_with_validation(pitcher_logs)
    
def run_study():
    long_term_study()
    short_term_study_files()
    
if __name__ == '__main__':
    run_study()