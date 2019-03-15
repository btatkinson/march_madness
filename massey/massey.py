import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

from tqdm import tqdm

# ordinals
odf = pd.read_csv('../data/MasseyOrdinals.csv')

# systems = list(odf['SystemName'].unique())
# systems18 = []
# for system in systems:
#     sdf = odf.loc[odf['SystemName']==system]
#     system_years = list(sdf['Season'].unique())
#     if 2018 in system_years:
#         print(system)
#         systems18.append(system)
#
# print(len(systems18))
# print(systems18)

systems18 = ['SEL', 'AP', 'BIH', 'DUN', 'MAS', 'MOR', 'POM', 'RPI', 'SAG', 'USA',
'WLK', 'WOB', 'ARG', 'RTH', 'WOL', 'COL', 'DOL', 'CNG', 'DES', 'WIL', 'DOK', 'PIG',
'KPK', 'KRA', 'REW', 'STH', 'SPW', 'PGH', 'DC2', 'DCI', 'LMC', 'RT', 'RTP', 'BUR',
'BBT', '7OT', 'SFX', 'EBP', 'TPR', 'BLS', 'DII', 'TRP', 'LOG', 'SP', 'SPR', 'TRK',
'BWE', 'HAS', 'FSH', 'DAV', 'KPI', 'FAS', 'SMN', 'DDB', 'ESR', 'FMG', 'PRR', 'SMS',
'SGR', 'ZAM', 'JNG', 'YAG', 'MMG', 'BNT', 'WMV', 'COX', 'JJK', 'LAB', 'STM']

odf = odf.loc[odf['SystemName'].isin(systems18)]

seasons = [2014, 2015, 2016, 2017, 2018]
odf = odf.loc[odf['Season'].isin(seasons)]

# reg season
rs_df = pd.read_csv("../data/RegularSeasonDetailedResults.csv")
rs_df = rs_df[['Season', 'DayNum', 'WTeamID', 'LTeamID']]

# cut off earliest part of season, before some ranking systems ranked
rs_df = rs_df.loc[rs_df['DayNum']>=65]

tourn_df = pd.read_csv('../data/NCAATourneyCompactResults.csv')
tourn_df = tourn_df[['Season', 'DayNum', 'WTeamID', 'LTeamID']]

rs_df = pd.concat([rs_df, tourn_df])

# print("Compiling season long errors")
# error_tracker = []
# # establish best ranking systems
# for season in tqdm(seasons):
#     sub_df = rs_df[rs_df.Season==season]
#     sub_ord = odf[odf.Season==season]
#     yearly_systems = list(odf['SystemName'].unique())
#
#     # test on pom system
#     for system in yearly_systems:
#         sys_df = sub_ord[sub_ord.SystemName==system]
#
#         sea_df = pd.merge_asof(left=sub_df, right=sys_df,left_on=['DayNum'],
#         right_on=['RankingDayNum'], left_by=['WTeamID', 'Season'], right_by=['TeamID', 'Season'],
#         allow_exact_matches=False)
#
#         sea_df = pd.merge_asof(left=sea_df, right=sys_df,left_on=['DayNum'],
#         right_on=['RankingDayNum'], left_by=['LTeamID', 'Season'], right_by=['TeamID', 'Season'],
#         allow_exact_matches=False, suffixes=['W', 'L'])
#
#         sea_df = sea_df.drop(columns=['TeamIDW', 'TeamIDL'])
#
#         sea_df['Wrating'] = 100-4*np.log(sea_df['OrdinalRankW']+1)-sea_df['OrdinalRankW']/22
#         sea_df['Lrating'] = 100-4*np.log(sea_df['OrdinalRankL']+1)-sea_df['OrdinalRankL']/22
#         sea_df['Prob'] = 1/(1+10**((sea_df['Lrating']-sea_df['Wrating'])/15))
#         sea_df['Error'] = -np.log(sea_df['Prob'])
#
#         error_tracker_row = [season, system, sea_df['Error'].mean()]
#         error_tracker.append(error_tracker_row)
#
# error_df = pd.DataFrame(error_tracker,columns=['Season', 'System', 'Error'])
#
# by_sys = error_df.groupby('System')['Error'].mean()
# by_sys = by_sys.reset_index()
# by_sys = by_sys.sort_values(by="Error")
#
# # select top 5 or top 10 systems
# by_sys = by_sys.head(10)
# print(by_sys)
# sel_systems = list(by_sys.System.values)
# # which seasons do they have ratings for?
# for sys in sel_systems:
#     sdf = odf[odf.SystemName==sys]
#     print([sys, list(sdf.Season.unique())])

# i'm going to use top 5
sel_systems = ['TRP', 'DOK', 'POM', 'SAG', 'SP']


# narrow down historical dfs
odf = odf[odf.SystemName.isin(sel_systems)]


all_szns = None
for season in tqdm(seasons):
    sub_df = rs_df[rs_df.Season==season]
    sub_ord = odf[odf.Season==season]

    all_rnks = None
    # test on pom system
    for system in sel_systems:
        sys_df = sub_ord[sub_ord.SystemName==system]

        sea_df = pd.merge_asof(left=sub_df, right=sys_df,left_on=['DayNum'],
        right_on=['RankingDayNum'], left_by=['WTeamID', 'Season'], right_by=['TeamID', 'Season'],
        allow_exact_matches=False)

        sea_df = pd.merge_asof(left=sea_df, right=sys_df,left_on=['DayNum'],
        right_on=['RankingDayNum'], left_by=['LTeamID', 'Season'], right_by=['TeamID', 'Season'],
        allow_exact_matches=False, suffixes=['W', 'L'])

        sea_df['WRating'] = 100-4*np.log(sea_df['OrdinalRankW']+1)-sea_df['OrdinalRankW']/22
        sea_df['LRating'] = 100-4*np.log(sea_df['OrdinalRankL']+1)-sea_df['OrdinalRankL']/22
        sea_df['Prob'] = 1/(1+10**((sea_df['LRating']-sea_df['WRating'])/15))
        sea_df['Error'] = -np.log(sea_df['Prob'])

        sea_df = sea_df.drop(columns=['TeamIDW', 'TeamIDL', "SystemNameW", 'SystemNameL'])

        rename_cols = ['OrdinalRankW', 'OrdinalRankL', 'RankingDayNumW', 'RankingDayNumL', 'WRating', 'LRating', 'Prob', 'Error']
        for rc in rename_cols:
            new_col_name = system + '_' + rc
            sea_df = sea_df.rename(columns={rc: new_col_name})

        if all_rnks is not None:
            all_rnks = pd.merge(all_rnks, sea_df, left_on=["Season", "DayNum", "WTeamID", "LTeamID"],
            right_on=["Season", "DayNum", "WTeamID", "LTeamID"])
        else:
            all_rnks = sea_df

    if all_szns is not None:
        all_szns = pd.concat([all_szns, all_rnks])
    else:
        all_szns = all_rnks

probs = all_szns[['TRP_Prob', 'DOK_Prob', 'POM_Prob', 'SAG_Prob', 'SP_Prob']].copy()

all_szns['Avg Prob'] = probs.mean(axis=1)
probs['Avg Prob'] = probs.mean(axis=1)

corr = probs.corr()

all_szns['Avg_Prob_Error'] = -np.log(all_szns['Avg Prob'])

errors = all_szns[['TRP_Error', 'DOK_Error', 'POM_Error', 'SAG_Error', 'SP_Error', 'Avg_Prob_Error']]

errors_list = []
num_games = len(errors)
for col in list(errors):
    print(col, errors[col].sum()/num_games)

sub_df = pd.read_csv('../data/SampleSubmissionStage1.csv')

# get ranks
odf = pd.read_csv('../data/MasseyOrdinals.csv')

# systems
years = [2014, 2015, 2016, 2017, 2018]
systems = ['TRP', 'DOK', 'POM', 'SAG', 'SP']
odf = odf[odf.Season.isin(years)]
odf = odf[odf.SystemName.isin(systems)]
odf = odf.groupby(['SystemName','Season','TeamID']).last().reset_index()

odf['Rating']= 100-4*np.log(odf['OrdinalRank']+1)-odf['OrdinalRank']/22

df = odf.groupby(['Season','TeamID'])['Rating'].mean()

df = df.reset_index()

sub_df['Season'] = sub_df['ID'].map(lambda x: int(x.split('_')[0]))
sub_df['Team1'] = sub_df['ID'].map(lambda x: int(x.split('_')[1]))
sub_df['Team2'] = sub_df['ID'].map(lambda x: int(x.split('_')[2]))
sub_df = sub_df.merge(df[['Season','TeamID','Rating']],how='left',
                                    left_on = ['Season','Team1'], right_on = ['Season','TeamID'])
sub_df = sub_df.merge(df[['Season','TeamID','Rating']],how='left',
                                    left_on = ['Season','Team2'], right_on = ['Season','TeamID'],
                                   suffixes=['W','L'])
sub_df['Pred'] = 1/(1+10**((sub_df['RatingL']-sub_df['RatingW'])/15))
sub_df[['ID', 'Pred']].to_csv('submission.csv', index=False)
print(sub_df[['ID', 'Pred']].head())







# end
