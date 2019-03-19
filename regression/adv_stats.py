import numpy as np
import pandas as pd

import random

teams = pd.read_csv('../data/Stage2/Teams.csv')
sdf = pd.read_csv('../data/Stage2/RegularSeasonDetailedResults.csv')
tdf = pd.read_csv('../data/Stage2/NCAATourneyDetailedResults.csv')

print(list(sdf.Season.unique()))

teams = teams[['TeamID', 'TeamName']]
teams = teams.rename(columns={
    'TeamID':'WTeamID',
    'TeamName':'WTeamName'
})
sdf = pd.merge(left=sdf, right=teams, on=['WTeamID'])
tdf = pd.merge(left=tdf, right=teams, on=['WTeamID'])
teams = teams.rename(columns={
    'WTeamID':'LTeamID',
    'WTeamName':'LTeamName'
})
sdf = pd.merge(left=sdf, right=teams, on=['LTeamID'])
tdf = pd.merge(left=tdf, right=teams, on=['LTeamID'])


scols = sdf.columns.tolist()
tcols = tdf.columns.tolist()

scols.insert(0, scols.pop(scols.index('LTeamName')))
scols.insert(0, scols.pop(scols.index('WTeamName')))
tcols.insert(0, tcols.pop(tcols.index('LTeamName')))
tcols.insert(0, tcols.pop(tcols.index('WTeamName')))
sdf = sdf.reindex(columns=scols)
tdf = tdf.reindex(columns=tcols)

sdf['S'] = 1
tdf['S'] = 0

df = pd.concat([sdf,tdf])

###################
## Preprocessing ##
###################

df = df.reset_index(drop=True)
df = df.sort_values(by='DayNum')

# add cumulative wins
# df['W_wins'] = df.groupby('WTeamID')['WResult'].cumsum()
# df['L_losses'] = df.groupby('LTeamID')['WResult'].cumsum()
#
# # right now we don't have winner losses or loser wins. easier to do later
# df['W_losses'] = 0
# df['L_wins'] = 0

# import vegas
vegas = pd.read_csv('./master.csv')

df = pd.merge(left=df,right=vegas,left_on=['Season','DayNum','WTeamID','LTeamID'],
right_on=['Season', 'DayNum', 'WTeamID','LTeamID'])

# change data to one team a line

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

wdf['Result'] = 1
ldf['Result'] = 0

df = pd.concat([wdf,ldf], sort=True)

df = df.rename(columns={
    '_line':'Line'}
)

# print(df['Ast'].mean())
# print(df['TO'].mean())

# very important that you sort!!
df = df.reset_index(drop=True)
df = df.sort_values(by='DayNum')

# merge elo
elo_ratings = pd.read_csv('./elo_ratings.csv')
df = pd.merge(left=df,right=elo_ratings,on=['Season', 'DayNum', 'TeamID'])
elo_ratings = elo_ratings.rename(columns={
    'TeamID':'OppTeamID',
    'Elo':'OppElo',
    'Relo':'OppRelo'
})
df = pd.merge(left=df,right=elo_ratings,on=['Season', 'DayNum', 'OppTeamID'])

df['Played_Game'] = 1
df['GP'] = df.groupby(['Season','TeamID'])['Played_Game'].cumsum()
df['OppGP'] = df.groupby(['Season', 'OppTeamID'])['Played_Game'].cumsum()

df['Wins'] = df.groupby(['Season','TeamID'])['Result'].cumsum()
df['Losses'] = df['GP'] - df['Wins']

df['OppLosses'] = df.groupby(['Season', 'OppTeamID'])['Result'].cumsum()
df['OppWins'] = df['OppGP'] - df['OppLosses']

# verify that everything is working
df1 = df.loc[df['TeamID']==1222]
df1 = df1.loc[df1['OppTeamID']==1153]

df2 = df.loc[df['TeamID']==1153]
df2 = df2.loc[df2['OppTeamID']==1222]

print(df1)
print(df2)

df = df.drop(columns=['Played_Game'])

# find per game values

df['Poss'] = df['FGA'] + (df['FTA'] * 0.475) + df['TO'] - df['OR']
df['OppPoss'] = df['OppFGA'] + (df['OppFTA'] * 0.475) + df['OppTO'] - df['OppOR']

# per poss
stat_cols = ['Score','Ast','TO','Blk','Stl','OR','DR','PF']
for stat in stat_cols:
    nc = stat + 'PP'
    oppc = 'Opp' + stat
    noppc = 'Opp' + stat + 'PP'
    df[nc] = (df.groupby(['Season','TeamID'])[stat].cumsum())/(df.groupby(['Season','TeamID'])['Poss'].cumsum())
    df[noppc] = (df.groupby(['Season', 'OppTeamID'])[oppc].cumsum())/(df.groupby(['Season','OppTeamID'])['OppPoss'].cumsum())

# allowed per poss
stat_cols = ['Score','Ast','TO','Blk','Stl','OR','DR','PF']
for stat in stat_cols:
    nc = stat + 'APP'
    oppc = 'Opp' + stat
    noppc = 'Opp' + stat + 'APP'
    df[nc] = (df.groupby(['Season','TeamID'])[oppc].cumsum())/(df.groupby(['Season','TeamID'])['OppPoss'].cumsum())
    df[noppc] = (df.groupby(['Season', 'OppTeamID'])[stat].cumsum())/(df.groupby(['Season','OppTeamID'])['Poss'].cumsum())

# df['RebPP'] = df['ORPP'] + df['DRPP']
# df['RebAPP'] = df['ORAPP'] + df['DRAPP']
# df['OppRebPP'] = df['OppORPP'] + df['OppDRPP']
# df['OppRebAPP'] = df['OppORAPP'] + df['OppDRAPP']

# ball control
df['BC'] = df['AstPP']/df['TOPP']
df['BCA'] = df['AstAPP']/df['TOAPP']
df['OppBC'] = df['OppAstPP']/df['OppTOPP']
df['OppBCA'] = df['OppAstAPP']/df['OppTOAPP']

# disruptions
df['Disrupt'] = df['StlPP'] + 1.94 * df['BlkPP']
df['DisruptA'] = df['StlAPP'] + 1.94 * df['BlkAPP']
df['OppDisrupt'] = df['OppStlPP'] + 1.94 * df['OppBlkPP']
df['OppDisruptA'] = df['OppStlAPP'] + 1.94 * df['OppBlkAPP']

df['FGM2'] = df['FGM'] - df['FGM3']
df['FGA2'] = df['FGA'] - df['FGA3']
df['OppFGM2'] = df['OppFGM'] - df['OppFGM3']
df['OppFGA2'] = df['OppFGA'] - df['OppFGA3']
pct_cols = [['FGM2', 'FGA2'], ['FGM3', 'FGA3'], ['FTM', 'FTA']]
for pc in pct_cols:
    numer = pc[0]
    denom = pc[1]
    nc = numer+'%'
    opp_numer = 'Opp'+numer
    opp_denom = 'Opp'+denom
    opp_nc = 'Opp'+numer+'%'
    df[nc] = (df.groupby(['Season','TeamID'])[numer].cumsum())/(df.groupby(['Season','TeamID'])[denom].cumsum())
    df[opp_nc] = (df.groupby(['Season', 'OppTeamID'])[opp_numer].cumsum())/(df.groupby(['Season', 'OppTeamID'])[opp_denom].cumsum())

# pct allowed
pct_cols = [['FGM2', 'FGA2'], ['FGM3', 'FGA3'], ['FTM', 'FTA']]
for pc in pct_cols:
    numer = pc[0]
    denom = pc[1]
    nc = numer+'A%'
    opp_numer = 'Opp'+numer
    opp_denom = 'Opp'+denom
    opp_nc = 'Opp'+numer+'A%'
    df[nc] = (df.groupby(['Season','TeamID'])[opp_numer].cumsum())/(df.groupby(['Season','TeamID'])[opp_denom].cumsum())
    df[opp_nc] = (df.groupby(['Season', 'OppTeamID'])[numer].cumsum())/(df.groupby(['Season', 'OppTeamID'])[denom].cumsum())

df['Win%'] = df['Wins']/df['GP']
df['OppWin%'] = df['OppWins']/df['OppGP']

df = df.drop(columns=[
    # 'ORPP','DRPP','ORAPP','DRAPP',
    # 'OppORPP','OppDRPP','OppORAPP','OppDRAPP',
    'AstPP','AstAPP', 'TOPP', 'TOAPP',
    'OppAstPP','OppAstAPP', 'OppTOPP', 'OppTOAPP',
    'StlPP','StlAPP','BlkPP','BlkAPP',
    'OppStlPP','OppStlAPP','OppBlkPP','OppBlkAPP'
    ])

print(list(df))

df['Win%'] = df['Wins']/df['GP']
df['OppWin%'] = df['OppWins']/df['OppGP']

# # print(df[['TeamID', 'OppTeamID', 'FGM2%', 'OppFGM2%', 'FTM%', 'OppFTM%']])


features = ['ScorePP','ScoreAPP','BC','BCA','Disrupt','DisruptA',
'FGM2%','FGM3%','FTM%','FGM2A%','FGM3A%','Win%',
'PFPP','PFAPP','ORPP','DRPP','ORAPP','DRAPP','Elo']

opp_features = []
for feature in features:
    opp_features.append(('Opp'+feature))


df[features] = df.groupby(['Season','TeamID'])[features].shift()
df[opp_features] = df.groupby(['Season', 'OppTeamID'])[opp_features].shift()

features = features + opp_features

features.extend(['Loc', 'DayNum', 'Line', 'Result'])

def loc2int(x):
    return loc_map[x]

loc_map = {
    'H':1,
    'A':0,
    'N':0.5
}
df['Loc'] = df['Loc'].apply(lambda x: change_loc(x))

print("You are using " + str(len(features)-1) + " features in your model.")

df_input = df[features]
df_input = df_input[df_input.DayNum>=25]

b4drop = len(df_input)
df_input = df_input.dropna()
aftdrop = len(df_input)
print("There were " + str(b4drop-aftdrop) + " rows with NaNs dropped")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# option 1
y = df_input.pop('Result').values
X = df_input.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=22)

# option 2
# X_train = df_input.loc[df_input.S==1]
# X_test = df_input.loc[df_input.S==0]
#
# y_train = X_train.pop('Result').values
# y_test = X_test.pop('Result').values
# X_train = X_train.values
# X_test = X_test.values

print("Fitting...")
clf = LogisticRegression(random_state=22).fit(X_train, y_train)

print("Predicting...")
y_pred = clf.predict_proba(X_test)

print(log_loss(y_test,y_pred))
print("Your score is " + str(np.round(clf.score(X,y)*100,2))+" %")
print("Best benchmark is 74.15 %, Erik Forseth")


#shift one








# end
