import numpy as np
import pandas as pd

############################
## Baseline Score: .54243 ##
############################

teams = pd.read_csv('../data/Stage2/Teams.csv')
sdf = pd.read_csv('../data/Stage2/RegularSeasonDetailedResults.csv')
tdf = pd.read_csv('../data/Stage2/NCAATourneyDetailedResults.csv')

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

# very important that you sort!!
df = df.reset_index(drop=True)
df = df.sort_values(by='DayNum')

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

stat_cols = ['Ast', 'Blk', 'DR', 'OR', 'PF', 'Score', 'Stl', 'TO']
for stat in stat_cols:
    nc = stat + 'PG'
    oppc = 'Opp' + stat
    noppc = 'Opp' + stat + 'PG'
    df[nc] = (df.groupby(['Season','TeamID'])[stat].cumsum())/df['GP']
    df[noppc] = (df.groupby(['Season', 'OppTeamID'])[oppc].cumsum())/df['GP']

# allowed per game
stat_cols = ['Score']
for stat in stat_cols:
    nc = stat + 'APG'
    oppc = 'Opp' + stat
    noppc = 'Opp' + stat + 'APG'
    df[nc] = (df.groupby(['Season','TeamID'])[oppc].cumsum())/df['GP']
    df[noppc] = (df.groupby(['Season', 'OppTeamID'])[stat].cumsum())/df['GP']

# verify
# print(df[['TeamID', 'ScorePG', 'OppTeamID', 'OppScorePG']])

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

df['Win%'] = df['Wins']/df['GP']
df['OppWin%'] = df['OppWins']/df['OppGP']
# print(df[['TeamID', 'OppTeamID', 'FGM2%', 'OppFGM2%', 'FTM%', 'OppFTM%']])


features = ['AstPG', 'BlkPG', 'DRPG', 'ORPG', 'PFPG', 'ScorePG', 'StlPG', 'TOPG',
'FGM2%','FGM3%','FTM%', 'Win%', 'ScoreAPG']

opp_features = []
for feature in features:
    opp_features.append(('Opp'+feature))


df[features] = df.groupby(['Season','TeamID'])[features].shift()
df[opp_features] = df.groupby(['Season', 'OppTeamID'])[opp_features].shift()

features = features + opp_features

features.extend(['Loc', 'DayNum', 'Result'])

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


#shift one








# end
