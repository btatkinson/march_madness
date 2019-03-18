import numpy as np
import pandas as pd

from team import Team
from elo import Elo
from game import Game

from tqdm import tqdm

df = pd.read_csv('../data/RegularSeasonDetailedResults.csv')

# df = pd.concat([sdf, tdf])
df = df[df.Season >= 2014]
df = df.reset_index()
seasons = df.Season.unique()

# need rolling strength of schedule ratings
# https://kenpom.com/blog/ratings-methodology-update/

def init_ratings(team_directory,teams):

    for team_id in teams:
        team_obj = Team(team_id)
        team_directory[team_id] = team_obj

    return team_directory

def add_error(day, games, team_directory):
    elo_error = 0
    relo_error = 0
    teams = list(team_directory.values())
    for team in teams:
        elo_error += team.elo_error
        relo_error += team.relo_error
        games += team.gp
    elo_error /= games
    relo_error /= games
    return [day, elo_error, relo_error]


rating_log = []
for season in tqdm(seasons):

    sdf = df.loc[df['Season']==season]
    # get unique team IDs
    wteams = list(sdf['WTeamID'].unique())
    lteams = list(sdf['LTeamID'].unique())
    teams = list(set(wteams + lteams))

    # initialize ratings
    team_directory = {}
    team_directory = init_ratings(team_directory, teams)

    # iterate schedule
    sdf = sdf.sort_values(by='DayNum', ascending=True)
    sdf = sdf.reset_index()

    avg_errors = []
    for index,row in tqdm(sdf.iterrows()):

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

        wreb = row['WOR'] + row['WDR']
        lreb = row['LOR'] + row['LDR']

        margin = wscore-lscore
        reb_margin = wreb-lreb

        loc = row['WLoc']

        game = Game(team1, team2, loc, margin, reb_margin, games_played)

        team1,team2 = game.play_game()

        team_directory[wteam] = team1
        team_directory[lteam] = team2

        rl_team_1 = [season, row['DayNum'], team1.tid, team1.elo]
        rl_team_2 = [season, row['DayNum'], team2.tid, team2.elo]
        rating_log.append(rl_team_1)
        rating_log.append(rl_team_2)

rating_df = pd.DataFrame(rating_log, columns=['Season', 'DayNum', 'WTeamID', 'WElo'])
df = pd.merge(left=df,right=rating_df, on=['Season', 'DayNum', 'WTeamID'])
rating_df = rating_df.rename(columns={
    'WTeamID':'LTeamID',
    'WElo':'LElo'
})
df = pd.merge(left=df,right=rating_df, on=['Season', 'DayNum', 'LTeamID'])

df = df.reset_index(drop=True)
print(df.tail())

wcols = list(df)
lcols = list(df)

# change to one team a line
wdf = df[wcols]
ldf = df[lcols]

# reverse location for loser
def change_loc(x):
    x = loc_map[x]
    return x

loc_map = {
    'H':'A',
    'A':'H',
    'N':'N'
}
ldf['WLoc'] = ldf['WLoc'].apply(lambda x: change_loc(x))

# rename columns
for col in list(wdf):
    if 'W' in col and col != 'WLoc':
        wdf = wdf.rename(columns={
            col:col[1:]
        })
        ldf = ldf.rename(columns={
            col:'Opp'+col[1:]
        })
    if 'L' in col and col != 'WLoc':
        wdf = wdf.rename(columns={
            col:'Opp'+col[1:]
        })
        ldf = ldf.rename(columns={
            col:col[1:]
        })

wdf=wdf.rename(columns={
    'WLoc':'Loc'
})
ldf=ldf.rename(columns={
    'WLoc':'Loc',
})

# check if the columns are the same
# print(list(set(list(wdf))-set(list(ldf))))
# print(list(set(list(ldf))-set(list(wdf))))

wdf['Result'] = 1
ldf['Result'] = 0

df = pd.concat([wdf,ldf], sort=True)

df = df.sort_values(by='DayNum')

# create "calendar" of ratings for adj purposes
calendar = []

# new df
ndf = None

for season in tqdm(seasons):
    sdf = df[df.Season==season]
    sdf = sdf.sort_values(by='DayNum')
    sdf['Poss'] = sdf['FGA'] + sdf['FTA'] * 0.475 + sdf['TO'] - sdf['OR']
    sdf['OppPoss'] = sdf['OppFGA'] + sdf['OppFTA'] * 0.475 + sdf['OppTO'] - sdf['OppOR']
    sdf['RebTotal'] = sdf['OR'] + sdf['DR']
    sdf['OppRebTotal'] = sdf['OppOR'] + sdf['OppDR']

    sdf[['SeaPoss','SeaScore','SeaTO','SeaFGA','SeaFGM',
    'OppSeaPoss','OppSeaScore','OppSeaTO','OppSeaFGA','OppSeaFGM',
    'SeaAst','OppSeaAst',
    'SeaStl','OppSeaStl','SeaBlk','OppSeaBlk']] = sdf.groupby('TeamID')['Poss','Score','TO','FGA','FGM',
    'OppPoss','OppScore','OppTO','OppFGA','OppFGM',
    'Ast','OppAst','Stl','OppStl','Blk','OppBlk'].cumsum()

    # print(sdf[['TeamID', 'Ast', 'SeaAst', 'Stl', 'SeaStl']])

    # add season long ratings
    sdf['SeaORtg'] = (sdf['SeaScore']/sdf['SeaPoss'])
    sdf['SeaDRtg'] = (sdf['OppSeaScore']/sdf['OppSeaPoss'])

    # success rate
    sdf['SeaOSR'] = (sdf['SeaPoss'] - sdf['SeaTO'] - (sdf['SeaFGA']-sdf['SeaFGM']))/sdf['SeaPoss']
    sdf['SeaDSR'] = (sdf['OppSeaPoss'] - sdf['OppSeaTO'] - (sdf['OppSeaFGA']-sdf['OppSeaFGM']))/sdf['OppSeaPoss']

    sdf['BC'] = sdf['SeaAst']/sdf['SeaTO']
    sdf['BCA'] = sdf['OppSeaAst']/sdf['OppSeaTO']

    sdf['Disrupt'] = ((sdf['OppSeaTO'] + sdf['SeaStl'])/2 + sdf['SeaBlk'])/sdf['OppSeaPoss']
    sdf['DisruptA'] = ((sdf['SeaTO'] + sdf['OppSeaStl'])/2 + sdf['OppSeaBlk'])/sdf['SeaPoss']

    if ndf is None:
        ndf=sdf
    else:
        ndf = pd.concat([ndf,sdf])

    for index, row in sdf.iterrows():
        daynum = row['DayNum']
        teamid = row['TeamID']
        ortg = row['SeaORtg']
        drtg = row['SeaDRtg']
        osr = row['SeaOSR']
        dsr = row['SeaDSR']

        calendar_entry = [season, daynum, teamid, ortg, drtg, osr, dsr]
        calendar.append(calendar_entry)



df = ndf.copy()
ndf = None

calendar_df = pd.DataFrame(calendar, columns=['Season', 'DayNum', 'OppTeamID', 'OppSeaORtg', 'OppSeaDRtg', 'OppSeaOSR', 'OppSeaDSR'])

df = df[['Season', 'DayNum', 'OppTeamID', 'Ast', 'Blk', 'DR', 'FGA', 'FGA3', 'FGM', 'FGM3', 'FTA', 'FTM', 'Loc', 'NumOT', 'OR', 'OppAst',
'OppBlk', 'OppDR', 'OppFGA', 'OppFGA3', 'OppFGM', 'OppFGM3', 'OppFTA', 'OppFTM', 'OppOR', 'OppPF', 'OppScore', 'OppStl', 'OppTO', 'PF',
'Result', 'Score', 'Stl', 'TO', 'TeamID', 'SeaORtg', 'SeaDRtg', 'SeaOSR', 'SeaDSR', 'BC', 'BCA', 'Disrupt', 'DisruptA', 'Elo', 'OppElo']]

df = pd.merge(left=df, right=calendar_df, left_on=['Season', 'DayNum','OppTeamID'], right_on=['Season', 'DayNum','OppTeamID'])
# get adj
df = df.sort_values(by=['Season','DayNum'])
#
# print("grouping and merging...")


# df[['DRtgADJ','ORtgADJ','DSRADJ','OSRADJ']] = df.groupby(['Season','TeamID'])['OppSeaORtg', 'OppSeaDRtg', 'OppSeaOSR', 'OppSeaDSR'].expanding().mean().reset_index(drop=True)

# print("done")
#


# df['Adj_SeaORtg'] = df['SeaORtg'] + (1.032 - df['ORtgADJ'])
# df['Adj_SeaOSR'] = df['SeaOSR'] + (0.351126 - df['OSRADJ'])
# #
# df['Adj_SeaDRtg'] = df['SeaDRtg'] + (1.032 - df['DRtgADJ'])
# df['Adj_SeaDSR'] = df['SeaDSR'] + (0.351126 - df['DSRADJ'])


# change loc to int
def loc_to_int(x):
    x = loc_map[x]
    return int(x)

loc_map = {
    'H':1,
    'A':0,
    'N':0.5
}
df.loc[:,'Loc'] = df['Loc'].apply(lambda x: loc_to_int(x))

trn_input = df.groupby(['Season','TeamID'])['SeaORtg', 'SeaDRtg', 'SeaOSR', 'SeaDSR', 'Elo'].last().reset_index()
print(list(trn_input))
trn_input.to_csv('./trn_input.csv')

df[['SeaORtg', 'SeaDRtg', 'SeaOSR', 'SeaDSR', 'Elo']] = df.groupby('TeamID')['SeaORtg', 'SeaDRtg', 'SeaOSR', 'SeaDSR', 'Elo'].shift()
df[['OppSeaORtg', 'OppSeaDRtg', 'OppSeaOSR', 'OppSeaDSR', 'OppElo']] = df.groupby('TeamID')['OppSeaORtg', 'OppSeaDRtg', 'OppSeaOSR', 'OppSeaDSR','OppElo'].shift()

model_input = df[['Season', 'DayNum', 'TeamID', 'Loc', 'SeaORtg', 'SeaDRtg', 'SeaOSR', 'SeaDSR', 'Elo',
'OppSeaORtg', 'OppSeaDRtg', 'OppSeaOSR', 'OppSeaDSR', 'OppElo','Result']]

model_input = model_input[model_input.DayNum >= 0]

# print(model_input.head())
model_input.to_csv('./input.csv')






# separate into different team files
# for season in seasons:
#     sdf = df[df.Season==season]
#     # get team ids
#     team_ids = sdf.TeamID.unique()
#     for tid in team_ids:
#         team_df = sdf[sdf.TeamID==tid]
#         team_df = team_df.sort_values(by='DayNum')
#         path = './team_files/' + str(season) + '/' + str(tid) + '.csv'
#         team_df.to_csv(path, index=False)

# team_csv = pd.read_csv('./team_files/2018/1329.csv')
# print(list(team_csv))
#
# rolling_cols = ['Score', 'OppScore']
#
# for col in rolling_cols:
#     l10_name = 'L10_' + col
#     l20_name = 'L20_' + col
#     team_csv[l10_name] = team_csv[col].shift().ewm(span=10,min_periods=5).mean()
#     team_csv[l20_name] = team_csv[col].shift().ewm(span=20,min_periods=10).mean()
#
# print(team_csv[['TeamID', 'Score', 'OppScore', 'L10_Score', 'L20_Score', 'L10_OppScore', 'L20_OppScore']])

# rolling_cols = ['Ast', 'Blk', 'DR', 'FGA', 'FGA3', 'FGM', 'FGM3', 'FTA', 'FTM', 'Loc',
#  'OR', 'OppAst', 'OppBlk', 'OppDR', 'OppFGA', 'OppFGA3', 'OppFGM', 'OppFGM3',
#  'OppFTA', 'OppFTM', 'OppOR', 'OppPF', 'OppScore', 'OppStl', 'OppTO',
#  'PF', 'Score', 'Stl', 'TO']


# rolling_cols =
# print(team_csv)



# print("Blocks: ", df['Blk'].mean())
# print("Fouls: ", df['PF'].mean())
# print("Steals: ", df['Stl'].mean())
# print("Assists: ", df['Ast'].mean())

# rebound margins
# df['O_Total Reb'] = df['OR'] + df['OppDR']
# df['D_Total Reb'] = df['DR'] + df['OppOR']
#
# df['O_Reb%'] = df['OR']/df['O_Total Reb']
# df['D_Reb%'] = df['DR']/df['D_Total Reb']




###################
## Preprocessing ##
###################

# # possessions via kenpom method
# df['WPoss'] = df['WFGA'] + (df['WFTA'] * 0.475) + df['WTO'] - df['WOR']
# df['LPoss'] = df['LFGA'] + (df['LFTA'] * 0.475) + df['LTO'] - df['LOR']
#
# # add total rebounds
# df['WTotReb'] = df['WOR'] + df['WDR']
# df['LTotReb'] = df['LOR'] + df['LDR']
#
# # pct of total rebounds for each team
# df['WTRPct'] = df['WTotReb']/(df['WTotReb' ]+ df['LTotReb'])
# df['LTRPct'] = df['LTotReb']/(df['LTotReb'] + df['WTotReb'])
#
# #ORB pct
# df['W ORB%'] = df['WOR']/df['WTotReb']
# df['L ORB%'] = df['LOR']/df['LTotReb']
#
# # to avoid overfitting, drop some columns
# ## drop ##
# df = df.drop(columns=['WTotReb','WOR','WDR','LTotReb','LOR','LDR'])
#
# # pct points from each scoring method
# df['W3PCT'] = (df['WFGM3'] * 3)/df['WScore']
# df['W2PCT'] = ((df['WFGM'] - df['WFGM3']) * 2)/df['WScore']
# df['WFTPCT'] = df['WFTM']/df['WScore']
#
# df['L3PCT'] = (df['LFGM3'] * 3)/df['LScore']
# df['L2PCT'] = ((df['LFGM'] - df['LFGM3']) * 2)/df['LScore']
# df['LFTPCT'] = df['LFTM']/df['LScore']
#
# # points per possession (Offensive rating, mirrors d rating)
# df['W ORtg'] = df['WScore']/df['WPoss']
# df['L ORtg'] = df['LScore']/df['LPoss']
#
# # "ball control" - assist to turnover ratio
# df['W BC'] = df['WAst']/df['WTO']
# df['L BC'] = df['LAst']/df['LTO']
#
# # "disruptions" -- pct of opponent possessions that end in avg(turnovers, steals) or block
# df['W Disr'] = ((df['LTO'] + df['WStl'])/2) + df['WBlk']/df['LPoss']
# df['L Disr'] = ((df['WTO'] + df['LStl'])/2) + df['LBlk']/df['WPoss']
#
# # Defensive rating -- pct of opponent possessions that don't end in made basket
# df['W DRtg'] = (df['LTO'] + df['WBlk'] + (df['LFGA'] - df['LFGM']))/df['LPoss']
# df['L DRtg'] = (df['WTO'] + df['LBlk'] + (df['WFGA'] - df['WFGM']))/df['WPoss']
#
# print(list(df))
#
# # plan is to separate winner and loser into separate data rows
# # eliminate some opponent cols that we don't care about, e.g. most opponent features
#
# cols = ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM',
# 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM',
# 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'WPoss',
# 'LPoss', 'WTRPct', 'LTRPct', 'W ORB%', 'L ORB%', 'W3PCT', 'W2PCT', 'WFTPCT', 'L3PCT',
# 'L2PCT', 'LFTPCT', 'W ORtg', 'L ORtg', 'W BC', 'L BC', 'W Disr', 'L Disr', 'W DRtg',
# 'L DRtg']
#
# wdf = df[['Season', 'DayNum', 'WTeamID', 'WLoc', 'WTRPct',
# 'W3PCT', 'WFTPCT', 'W ORtg', 'W BC', 'W Disr', 'W DRtg']]
#
# ldf = df[['Season', 'DayNum', 'LTeamID', 'WLoc', 'LTRPct',
# 'L3PCT', 'LFTPCT', 'L ORtg', 'L BC', 'L Disr', 'L DRtg']]

#
# ['Season', 'DayNum', 'WTeamID', 'WLoc', 'WTRPct',
# 'W3PCT', 'WFTPCT', 'W ORtg', 'W BC', 'W Disr', 'W DRtg']
# # rename columns
# wdf = wdf.rename(columns={
#     'WTeamID':'TeamID',
#     ''
# })

# test on 2018
# df = df[df.Season==2018]
#
# print(len(df))






# end
