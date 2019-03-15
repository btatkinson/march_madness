import numpy as np
import pandas as pd
import math

from settings import *
from team import Team
from game import Game

from tqdm import tqdm

def init_ratings(team_directory,teams):

    for team_id in teams:
        team_obj = Team(team_id)
        team_directory[team_id] = team_obj

    return team_directory

def add_error(day, games, team_directory):
    games = 0
    wlm_error = 0
    elo_error = 0
    dim_error = 0
    mov_error = 0
    dimv_error = 0
    teams = list(team_directory.values())
    for team in teams:
        wlm_error += team.wlm_error
        elo_error += team.elo_error
        dim_error += team.dim_error
        mov_error += team.mov_error
        dimv_error += team.dimv_error
        games += team.gp
    wlm_error /= games
    elo_error /= games
    dim_error /= games
    mov_error /= games
    dimv_error /= games
    return [day, wlm_error, elo_error, dim_error, mov_error, dimv_error]

def run_history():

    # need to divide up history into individual season dataframes
    path = "../data/RegularSeasonDetailedResults.csv"
    complete_history = pd.read_csv(path)

    # get season list
    cols = ['Day', 'WLM Error', 'Elo Error', 'Dim Error', 'MOV Error', 'DimV Error']
    all_errors = pd.DataFrame(columns=cols)
    all_ratings = pd.DataFrame(columns=['Season', 'TID', 'Rating'])
    seasons = list(complete_history['Season'].unique())
    for season in seasons:
        # testing purposes
        # if season != 2017:
        #     continue

        # get season df
        season_df = complete_history.loc[complete_history['Season']==season]

        # get unique team IDs
        wteams = list(season_df['WTeamID'].unique())
        lteams = list(season_df['LTeamID'].unique())
        teams = list(set(wteams + lteams))

        # initialize ratings
        team_directory = {}
        team_directory = init_ratings(team_directory, teams)

        # iterate schedule
        season_df = season_df.sort_values(by='DayNum', ascending=True)
        season_df = season_df.reset_index()

        avg_errors = []
        for index,row in tqdm(season_df.iterrows()):

            wteam = row['WTeamID']
            lteam = row['LTeamID']

            daynum = row['DayNum']

            # need games_played num to get K value
            team1 = team_directory[wteam]
            team2 = team_directory[lteam]
            games_played = (team1.gp + team2.gp)/2

            if index == 0:
                last_day = row['DayNum']
            else:
                if row['DayNum'] > last_day:
                    new_day = True
                    error_row = add_error(last_day, (index+1), team_directory)
                    avg_errors.append(error_row)
                    last_day = row['DayNum']
                else:
                    new_day = False

            wscore = row['WScore']
            lscore = row['LScore']

            margin = wscore-lscore

            loc = row['WLoc']

            game = Game(team1, team2, loc, margin, games_played)

            team1,team2 = game.play_game()

            team_directory[wteam] = team1
            team_directory[lteam] = team2

        s_error_df = pd.DataFrame(avg_errors, columns=cols)
        all_errors = pd.concat([all_errors,s_error_df])

        dimv_ratings = []
        for team in list(team_directory.values()):
            dimv_ratings.append([season,team.tid,team.dimv])
        ratings_df = pd.DataFrame(dimv_ratings,columns=['Season', 'TID', 'Rating'])
        ratings_df = ratings_df.sort_values(by='Rating', ascending=False)
        all_ratings = pd.concat([all_ratings, ratings_df])

    print(all_errors.tail())
    all_errors.to_csv('sub_errors.csv')

    print(all_ratings)
    all_ratings.to_csv('rate.csv')

    return


def create_submission():
    sub_df = pd.read_csv('../data/SampleSubmissionStage1.csv')
    ratings = pd.read_csv('rate.csv')

    sub_df['Season'] = sub_df['ID'].map(lambda x: int(x.split('_')[0]))
    sub_df['Team1'] = sub_df['ID'].map(lambda x: int(x.split('_')[1]))
    sub_df['Team2'] = sub_df['ID'].map(lambda x: int(x.split('_')[2]))
    sub_df = sub_df.merge(ratings[['Season','TID','Rating']],how='left',
                                        left_on = ['Season','Team1'], right_on = ['Season','TID'])
    sub_df = sub_df.merge(ratings[['Season','TID','Rating']],how='left',
                                        left_on = ['Season','Team2'], right_on = ['Season','TID'],
                                       suffixes=['W','L'])
    sub_df['Pred'] = 1/(1+10**((sub_df['RatingL']-sub_df['RatingW'])/ 400))
    sub_df[['ID', 'Pred']].to_csv('submission.csv', index=False)
    print(sub_df[['ID', 'Pred']].head())
    return





if __name__ == "__main__":
    run_history()

    create_submission()







# end
