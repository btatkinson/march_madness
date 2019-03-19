import pandas as pd
import numpy as np
import math

from datetime import datetime

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%m/%d/%Y")
    d2 = datetime.strptime(d2, "%m/%d/%Y")
    return 154-abs((d2 - d1).days)

seasons = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

# aggregate
agg_df = None
for season in seasons:
    strseason = str(season)
    season_col = season + 1
    year = strseason[-2:]
    season_vegas = pd.read_csv('../data/Vegas/ncaabb'+year+'.csv')
    season_vegas['Season'] = season_col
    season_vegas = season_vegas.dropna(how='all')
    if season == 2017:
        last_day = '04/02/2018'
    elif season == 2018:
        last_day = '04/08/2019'
    else:
        last_day = season_vegas.tail(1).date.values[0]
    season_vegas['date'] = season_vegas['date'].astype(str)
    season_vegas['date'] = season_vegas['date'].fillna(0)
    season_vegas['date'] = season_vegas['date'].replace('nan',0)
    season_vegas['DayNum'] = season_vegas['date'].apply(lambda x: 0 if x==0 else days_between(last_day, x))
    if agg_df is None:
        agg_df = season_vegas
    else:
        agg_df = pd.concat([agg_df,season_vegas])

# fill na on line
agg_df['line'] = agg_df['line'].fillna(agg_df['lineopen'])
agg_df['line'] = agg_df['line'].fillna(agg_df['linesag'])
agg_df['line'] = agg_df['line'].fillna(agg_df['lineavg'])

print(agg_df.line.isna().sum())

# convert names to ids
agg_df.home = agg_df.home.str.lower()
agg_df.road = agg_df.road.str.lower()

team_names = pd.read_csv('../data/TeamSpellings.csv',encoding="ISO-8859-1")

unique_names = set(agg_df.home.unique())
our_names = set(agg_df.home.unique())
print("missing names: ",list(unique_names-our_names))
team_names = team_names.rename(columns={
'TeamID':'HTeamID'
})
agg_df = pd.merge(left=agg_df,right=team_names,left_on=['home'], right_on=['TeamNameSpelling'])
team_names = team_names.rename(columns={
'HTeamID':'ATeamID'
})
agg_df = pd.merge(left=agg_df,right=team_names,left_on=['road'], right_on=['TeamNameSpelling'])

agg_df.line = agg_df.line.replace('.',0)
agg_df.line = agg_df.line.astype(float)

agg_df['Result'] = np.where(agg_df['hscore']>agg_df['rscore'], 1, 0)
agg_df['WTeamID'] = np.where(agg_df['Result']==1, agg_df['HTeamID'], agg_df['ATeamID'])
agg_df['LTeamID'] = np.where(agg_df['Result']==0, agg_df['HTeamID'], agg_df['ATeamID'])
agg_df['W_line'] = np.where(agg_df['Result']==1, agg_df['line'], agg_df['line']*-1)
agg_df['L_line'] = np.where(agg_df['Result']==0, agg_df['line'], agg_df['line']*-1)

agg_df = agg_df[['Season','DayNum','WTeamID','LTeamID','W_line','L_line']]

agg_df = agg_df.sort_values(by=['Season','DayNum'])

agg_df.to_csv('./master.csv',index=False)

print(agg_df)


#end
