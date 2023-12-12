# CS184A_Final_Project

This is the repository of the CS 184A project titled "Predicting Ulnar Collateral Ligament (UCL) Injuries Among Major League Baseball Players" By Connor O'Rourke and Hanvish Vdaygiri

[Final Project Report](https://haxor48.github.io/predicting_ucl_injuries/)

## Setup

To run our code, you will need to follow the following steps:
1) Download the repository attatched, as there's some needed files including the .py file
2) Install the following modules:
   - Numpy (pip install numpy)
   - pybaseball (pip install pybaseball)
   - pandas (pip install pandas)
   - Scikit-learn (pip install -U scikit-learn)
   - Tensorflow (pip install tensorflow)
   - matplotlib (python -m pip install -U matplotlib)
3) Run the file final_project_184a.py

## Method documentation

### Helper Methods (No real help to a user)
- **get_injury_year:** Given a date returns the year the injury occurred from
- **remove_accents:** Removes accents from player names (Baseball Savant has accents while Fangraphs doesn't)
- **fix_name:** Turns a name from last,first to first last and removes accents
- **merge_sections:** Merges the spin rate table and season-long Fangraphs table together
- **aggregate_merged:** Removes some pitch-by-pitch stats for proper averages that works better with the models
- **get_X_Y:** Removes labels from merged to be used in test splits
- **split_train_test:** Does the 80:20 train test split
- **injured_year:** Outputs where a player was injured in a given season
- **save_pitchers:** Saves the game-by-game pitcher data
- **fix_injuries:** Fixes the labels on the game-by-game data
- **first_occurence:** Returns the first row a certain pitch appears in the dataframe
- **calc_difference:** Returns a dataframe of the difference between the metrics of 2 games
- **save_delta:** Saves the game-by-game differences
- **drop_na:** Removes columns where the values are na (Issue with the Baseball Savant API)

### Usuable Methods (Will give noticeable output for the user)
- **build_ucl_injuries:** Returns a dataframe with a list of every UCL injury
- **add_ucl_injuries_to_table:** Returns a dataframe of Fangraphs pitcher data with lables signfying a UCL injury occurring, where the totals are for each season from 2018-2023
- **get_spin:** Returns a dataframe of the season-long spin rate averages per pitch for each pitcher from 2018-2023
- **build_and_clean_long_term:** Returns a dataframe that's the merged and pre-processed data to be used in the models
- **get_best_k:** Given test splits, and a list of possible k values, plots loss vs the k values on a log scale
- **knn:** Given test splits, runs the K-Nearest Neighbors algorithm
- **logistic_regression:** Given test splits, runs a logistic regression classifier on the data
- **mlp:** Given test splits, runs a Muli-Layered Perceptron on the data
- **cnn_with_validation:** Given the labels and data, runs a Convolutional Neural Network and does validation on the model
- **get_game_codes:** Returns a list of all the codes of games from 2022-2023
- **update_game_by_game:** Returns a dictionary of dataframes of the game-by-game pitch data for each pitcher (Note: takes a long time to run)
- **load_pitchers:** Loads the game-by-game pitcher data from a file
- **load_delta:** Loads the differences in game-by-game pitcher data from a file
- **update_delta:** Given the game-by-game data, returns a dictionary of dataframes of the changes in metrics between games for each pitcher
- **rnn_with_validation:** Given the dictionary of pitcher data, runs a Recurrent Neural Network and does validation
- **long_term_study:** Compiles all the long-term data and runs each algorithm/model
- **short_term_study_files:** Compiles the short-term data from local files and runs the RNN
- **short_term_study_api:** Builds the short-term data from the start and runs the RNN (Note: Will take a long time (3hrs+))
- **run_study:** Runs both studies. You can specify true to run from the api, but it defaults to false
