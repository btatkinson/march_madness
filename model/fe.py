import pandas as pd
import numpy as np

sdf = pd.read_csv('./sdf_raw.csv')

# tournament score
# career tournament score
sdf['CoachCTS'] = 6*sdf['CChmps'] + 4*sdf['CFFs'] + 2*sdf['CS16s'] + sdf['CTourns']
sdf = sdf.drop(columns=['CChmps','CFFs','CS16s','CTourns'])
# career tournament score
sdf['OppCoachCTS'] = 6*sdf['OppCChmps'] + 4*sdf['OppCFFs'] + 2*sdf['OppCS16s'] + sdf['OppCTourns']
sdf = sdf.drop(columns=['OppCChmps','OppCFFs','OppCS16s','OppCTourns'])
# Coach school tournament score
sdf['CoachSTS'] = 6*sdf['SChmps'] + 4*sdf['SFFs'] + 2*sdf['SS16s'] + sdf['STourns']
sdf = sdf.drop(columns=['SChmps','SFFs','SS16s','STourns'])
# career tournament score
sdf['OppCoachSTS'] = 6*sdf['OppSChmps'] + 4*sdf['OppSFFs'] + 2*sdf['OppSS16s'] + sdf['OppSTourns']
sdf = sdf.drop(columns=['OppSChmps','OppSFFs','OppSS16s','OppSTourns'])

sdf = sdf.rename(columns={
    'SchoolWins':'CoachWAS',
    'SchoolLosses':'CoachLAS',
    'OppSchoolWins':'OppCoachWAS',
    'OppSchoolLosses':'OppCoachLAS',
    'CareerWins':'CoachCWs',
    'CareerLosses':'CoachCLs',
    'OppCareerWins':'OppCoachCWs',
    'OppCareerLosses':'OppCoachCLs',
})

sdf = sdf.drop(columns=['School','Coach'])

# important to sort by daynum!
sdf = sdf.sort_values(by='DayNum')

sdf['Played_Game'] = 1
sdf['GP'] = sdf.groupby(['Season','TeamID'])['Played_Game'].cumsum()
sdf['OppGP'] = sdf.groupby(['Season', 'OppTeamID'])['Played_Game'].cumsum()

sdf['Wins'] = sdf.groupby(['Season','TeamID'])['Result'].cumsum()
sdf['Losses'] = sdf['GP'] - sdf['Wins']

sdf['OppWins'] = sdf.groupby(['Season', 'OppTeamID'])['Result'].cumsum()
sdf['OppLosses'] = sdf['OppGP'] - sdf['OppWins']

sdf = sdf.drop(columns=['Played_Game'])
# test
# df1 = sdf.loc[sdf['TeamID']==1222]
# df1 = df1.loc[df1['OppTeamID']==1153]
#
# df2 = sdf.loc[sdf['TeamID']==1153]
# df2 = df2.loc[df2['OppTeamID']==1222]
#
# print(df1[['Season','DayNum','TeamID','OppTeamID','GP','Wins','Losses','OppGP','OppLosses','OppWins']])
#
# df = df.drop(columns=['Played_Game'])

# add the wins from the season
sdf['CoachCWs'] += sdf['Wins']
sdf['CoachCLs'] += sdf['Losses']
sdf['OppCoachCWs'] += sdf['OppWins']
sdf['OppCoachCLs'] += sdf['OppLosses']
sdf['CoachWAS'] += sdf['Wins']
sdf['CoachLAS'] += sdf['Losses']
sdf['OppCoachWAS'] += sdf['OppWins']
sdf['OppCoachLAS'] += sdf['OppLosses']

sdf['Poss'] = sdf['FGA'] + (sdf['FTA'] * 0.475) + sdf['TO'] - sdf['OR']
sdf['OppPoss'] = sdf['OppFGA'] + (sdf['OppFTA'] * 0.475) + sdf['OppTO'] - sdf['OppOR']

sdf['MinPlayed'] = 40
sdf['MinPlayed'] += (sdf['NumOT'] * 5)
sdf['OppMinPlayed'] = 40
sdf['OppMinPlayed'] += (sdf['NumOT'] * 5)

sdf['FGM2'] = sdf['FGM'] - sdf['FGM3']
sdf['FGA2'] = sdf['FGA'] - sdf['FGA3']
sdf['OppFGM2'] = sdf['OppFGM'] - sdf['OppFGM3']
sdf['OppFGA2'] = sdf['OppFGA'] - sdf['OppFGA3']

stat_cols = ['Score','Ast','TO','Blk','Stl','OR','DR','PF','MinPlayed','FGM','FGM2','FTA','FGA']
for stat in stat_cols:
    nc = stat + 'PP'
    oppc = 'Opp' + stat
    noppc = 'Opp' + stat + 'PP'
    sdf[nc] = (sdf.groupby(['Season','TeamID'])[stat].cumsum())/(sdf.groupby(['Season','TeamID'])['Poss'].cumsum())
    sdf[noppc] = (sdf.groupby(['Season', 'OppTeamID'])[oppc].cumsum())/(sdf.groupby(['Season','OppTeamID'])['OppPoss'].cumsum())
    if stat == 'MinPlayed':
        sdf[nc] = 1/sdf[nc]
        sdf[noppc] = 1/sdf[noppc]

sdf = sdf.rename(columns={
    'MinPlayedPP':'Tempo',
    'OppMinPlayedPP':'OppTempo',
})
sdf = sdf.drop(columns=['MinPlayed','OppMinPlayed'])

stat_cols = ['Score','Ast','TO','Blk','Stl','OR','DR','PF','FTA','FGM','FGM2','FGA']
for stat in stat_cols:
    nc = stat + 'APP'
    oppc = 'Opp' + stat
    noppc = 'Opp' + stat + 'APP'
    sdf[nc] = (sdf.groupby(['Season','TeamID'])[oppc].cumsum())/(sdf.groupby(['Season','TeamID'])['OppPoss'].cumsum())
    sdf[noppc] = (sdf.groupby(['Season', 'OppTeamID'])[stat].cumsum())/(sdf.groupby(['Season','OppTeamID'])['Poss'].cumsum())

# Pct Cols
sdf['Win%'] = sdf['Wins']/sdf['GP']
sdf['OppWin%'] = sdf['OppWins']/sdf['OppGP']

sdf['CoachGPAS'] = sdf['CoachWAS'] + sdf['CoachLAS']
sdf['CoachWPctS'] = sdf['CoachWAS'] / sdf['CoachGPAS']
sdf['OppCoachGPAS'] = sdf['OppCoachWAS'] + sdf['OppCoachLAS']
sdf['OppCoachWPctS'] = sdf['OppCoachWAS'] / sdf['OppCoachGPAS']

pct_cols = [['FGM2', 'FGA2'], ['FGM3', 'FGA3'], ['FTM', 'FTA'],['FGM','FGA']]
for pc in pct_cols:
    numer = pc[0]
    denom = pc[1]
    nc = numer+'%'
    opp_numer = 'Opp'+numer
    opp_denom = 'Opp'+denom
    opp_nc = 'Opp'+numer+'%'
    sdf[nc] = (sdf.groupby(['Season','TeamID'])[numer].cumsum())/(sdf.groupby(['Season','TeamID'])[denom].cumsum())
    sdf[opp_nc] = (sdf.groupby(['Season', 'OppTeamID'])[opp_numer].cumsum())/(sdf.groupby(['Season', 'OppTeamID'])[opp_denom].cumsum())

# pct allowed
pct_cols = [['FGM2', 'FGA2'], ['FGM3', 'FGA3'], ['FTM', 'FTA'],['FGM','FGA']]
for pc in pct_cols:
    numer = pc[0]
    denom = pc[1]
    nc = numer+'A%'
    opp_numer = 'Opp'+numer
    opp_denom = 'Opp'+denom
    opp_nc = 'Opp'+numer+'A%'
    sdf[nc] = (sdf.groupby(['Season','TeamID'])[opp_numer].cumsum())/(sdf.groupby(['Season','TeamID'])[opp_denom].cumsum())
    sdf[opp_nc] = (sdf.groupby(['Season', 'OppTeamID'])[numer].cumsum())/(sdf.groupby(['Season', 'OppTeamID'])[denom].cumsum())

sdf['TR%'] = (sdf['ORPP'] + sdf['DRPP'])/(sdf['ORPP'] + sdf['DRPP'] + sdf['ORAPP'] + sdf['DRAPP'])
sdf['OppTR%'] = (sdf['OppORPP'] + sdf['OppDRPP'])/(sdf['OppORPP'] + sdf['OppDRPP'] + sdf['OppORAPP'] + sdf['OppDRAPP'])

sdf['Ast%'] = sdf['AstPP']/sdf['FGM2PP']
sdf['OppAst%'] = sdf['OppAstPP']/sdf['OppFGM2PP']

# std cols
# print(sdf.groupby(['Season','TeamID'])['Score'])

# ball control
sdf['BC'] = sdf['AstPP']/sdf['TOPP']
sdf['BCA'] = sdf['AstAPP']/sdf['TOAPP']
sdf['OppBC'] = sdf['OppAstPP']/sdf['OppTOPP']
sdf['OppBCA'] = sdf['OppAstAPP']/sdf['OppTOAPP']

# disruptions
sdf['Disrupt'] = sdf['StlPP'] + 1.94 * sdf['BlkPP']
sdf['DisruptA'] = sdf['StlAPP'] + 1.94 * sdf['BlkAPP']
sdf['OppDisrupt'] = sdf['OppStlPP'] + 1.94 * sdf['OppBlkPP']
sdf['OppDisruptA'] = sdf['OppStlAPP'] + 1.94 * sdf['OppBlkAPP']

features = ['ScorePP','ScoreAPP',
'TOPP','BlkPP','StlPP','ORPP','DRPP','PFPP',
'TOAPP','BlkAPP','StlAPP','ORAPP','DRAPP','PFAPP',
'Win%','CoachWPctS',
# 'CoachWAS','CoachLAS',
'BC','BCA','Disrupt','DisruptA',
'FGM2%','Elo','TR%', 'Ast%'
# 'PFPP','PFAPP','ORPP','DRPP','ORAPP','DRAPP','Elo'
]

# sdf['ScoringMargin'] = sdf['Score'] - sdf['OppScore']
# a=sdf.groupby(['Season','TeamID'])['ScoringMargin'].agg(('sum', 'count'))
# # a = a.reindex(pd.MultiIndex.from_product([['1', '2'], range(1,7)], names=['userid', 'week']))
# b = a.groupby(level=0).cumsum().groupby(level=0).shift(1)
# b['avg_sm'] = b['sum'] / b['count']
# c = b.reset_index().drop(['count', 'sum'], axis=1)
# d = c.groupby('TeamID').fillna(method='ffill')
# d['TeamID'] = c['TeamID']
# d = d[['TeamID', 'Season', 'avg_sm']]
# print(d)
# raise ValueError


opp_features = []
for feature in features:
    opp_features.append(('Opp'+feature))

trn_input = sdf.copy()
sdf[features] = sdf.groupby(['Season','TeamID'])[features].shift()
sdf[opp_features] = sdf.groupby(['Season', 'OppTeamID'])[opp_features].shift()

features = features + opp_features
tfeatures = features + ['Season','DayNum','TeamID']
trn_input = sdf[tfeatures]
features.extend(['Loc', 'DayNum', 'Result'])

sdf_input = sdf[features]
sdf_input = sdf_input[sdf_input.DayNum>=50]

b4drop = len(sdf_input)
sdf_input = sdf_input.dropna()
aftdrop = len(sdf_input)

print(sdf_input.head())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# option 1
y = sdf_input.pop('Result')
X = sdf_input
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=22)

print("Fitting...")
clf = LogisticRegression(solver='lbfgs',max_iter=100)

clf.fit(X_train, y_train)
# print("Predicting...")
y_pred = clf.predict_proba(X_test)
print(log_loss(y_test,y_pred))
print("Your score is " + str(np.round(clf.score(X_test,y_test)*100,2))+" %")
print("Best benchmark is 74.15 %, Erik Forseth")

print("Loading tournament data..")

print(len(trn_input))
print(list(trn_input))
trn_input = trn_input.sort_values(by='DayNum')
trn_input = trn_input.groupby(['Season','TeamID']).last().reset_index()
print(trn_input)


# tdf = pd.read_csv('../data/Stage2/NCAATourneyCompactResults.csv')
sub_df = pd.read_csv('../data/SampleSubmissionStage1.csv')

sub_df['Loc'] = 0.5
sub_df['DayNum'] = 133

sub_df['Season'] = sub_df['ID'].map(lambda x: int(x.split('_')[0]))
sub_df['Team1'] = sub_df['ID'].map(lambda x: int(x.split('_')[1]))
sub_df['Team2'] = sub_df['ID'].map(lambda x: int(x.split('_')[2]))
sub_df = sub_df.merge(trn_input,how='left',
                                    left_on = ['Season','Team1'], right_on = ['Season','TeamID'])
sub_df = sub_df.merge(trn_input,how='left',
                                    left_on = ['Season','Team2'], right_on = ['Season','TeamID'],
                                   suffixes=['W','L'])

print(sub_df.head())
features = ['ScorePP','ScoreAPP',
'TOPP','BlkPP','StlPP','ORPP','DRPP','PFPP',
'TOAPP','BlkAPP','StlAPP','ORAPP','DRAPP','PFAPP',
'Win%','CoachWPctS',
# 'CoachWAS','CoachLAS',
'BC','BCA','Disrupt','DisruptA',
'FGM2%','Elo','TR%', 'Ast%'
# 'PFPP','PFAPP','ORPP','DRPP','ORAPP','DRAPP','Elo'
]
t1_features = []
t2_features = []
for feature in features:
    ft1 = feature + 'W'
    t1_features.append(ft1)
for feature in features:
    ft2 = feature + 'L'
    t2_features.append(ft2)
trn_features = t1_features + t2_features
trn_features.extend(['Loc','DayNum'])
sub_df = sub_df.drop(columns=['TeamIDL','TeamIDW'])
predict_df = sub_df[trn_features]
sub_df['Pred'] = clf.predict_proba(predict_df)
sub_df[['ID', 'Pred']].to_csv('submission.csv', index=False)
print(sub_df[['ID', 'Pred']].head())
# sub_df = pd.merge()

# print(tdf.head())


# end
