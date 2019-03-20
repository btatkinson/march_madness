import numpy as np
import pandas as pd

def fix_scd(scd,season):
    season_wins = scd.iloc[1:,4].astype(float)
    season_losses = scd.iloc[1:,4].astype(float)

    # career at school
    cas_wins = scd.iloc[1:,10].astype(float)
    cas_losses = scd.iloc[1:,11].astype(float)

    cas_wins -= season_wins
    cas_losses -= season_losses

    car_wins = scd.iloc[1:,17].astype(float)
    car_losses = scd.iloc[1:,18].astype(float)

    car_wins -= season_wins
    car_losses -= season_losses

    cas_trns = np.where(scd.iloc[1:,8]=='Lost First Round', scd.iloc[1:,13].astype(float)-1, scd.iloc[1:,13])
    cas_trns = np.where(scd.iloc[1:,8]=='Lost Second Round', cas_trns.astype(float)-1, cas_trns)
    cas_trns = np.where(scd.iloc[1:,8]=='Lost Third Round', cas_trns.astype(float)-1, cas_trns)
    cas_trns = np.where(scd.iloc[1:,8]=='Lost Regional Semifinal', cas_trns.astype(float)-1, cas_trns)
    cas_trns = np.where(scd.iloc[1:,8]=='Lost Regional Final', cas_trns.astype(float)-1, cas_trns)
    cas_trns = np.where(scd.iloc[1:,8]=='Lost National Semifinal', cas_trns.astype(float)-1, cas_trns)
    cas_trns = np.where(scd.iloc[1:,8]=='Lost National Final', cas_trns.astype(float)-1, cas_trns)
    cas_trns = np.where(scd.iloc[1:,8]=='Won National Final', cas_trns.astype(float)-1, cas_trns)

    scd.iloc[1:,14] = scd.iloc[1:,14].replace('S16',0)
    cas_s16s = np.where(scd.iloc[1:,8]=='Lost Third Round', scd.iloc[1:,14].astype(float)-1, scd.iloc[1:,14])
    cas_s16s = np.where(scd.iloc[1:,8]=='Lost Regional Semifinal', cas_s16s.astype(float)-1, cas_s16s)
    cas_s16s = np.where(scd.iloc[1:,8]=='Lost Regional Final', cas_s16s.astype(float)-1, cas_s16s)
    cas_s16s = np.where(scd.iloc[1:,8]=='Lost National Semifinal', cas_s16s.astype(float)-1, cas_s16s)
    cas_s16s = np.where(scd.iloc[1:,8]=='Lost National Final', cas_s16s.astype(float)-1, cas_s16s)
    cas_s16s = np.where(scd.iloc[1:,8]=='Won National Final', cas_s16s.astype(float)-1, cas_s16s)

    scd.iloc[1:,15] = scd.iloc[1:,15].replace('FF',0)
    cas_ffs = np.where(scd.iloc[1:,8]=='Lost National Semifinal', scd.iloc[1:,15].astype(float)-1, scd.iloc[1:,15])
    cas_ffs = np.where(scd.iloc[1:,8]=='Lost National Final', cas_ffs.astype(float)-1, cas_ffs)
    cas_ffs = np.where(scd.iloc[1:,8]=='Won National Final', cas_ffs.astype(float)-1, cas_ffs)

    scd.iloc[1:,16] = scd.iloc[1:,16].replace('Chmp',0)
    cas_chmps = np.where(scd.iloc[1:,8]=='Won National Final', scd.iloc[1:,16].astype(float)-1, scd.iloc[1:,16])


    car_trns = np.where(scd.iloc[1:,8]=='Lost First Round', scd.iloc[1:,20].astype(float)-1, scd.iloc[1:,20])
    car_trns = np.where(scd.iloc[1:,8]=='Lost Second Round', car_trns.astype(float)-1, car_trns)
    car_trns = np.where(scd.iloc[1:,8]=='Lost Third Round', car_trns.astype(float)-1, car_trns)
    car_trns = np.where(scd.iloc[1:,8]=='Lost Regional Semifinal', car_trns.astype(float)-1, car_trns)
    car_trns = np.where(scd.iloc[1:,8]=='Lost Regional Final', car_trns.astype(float)-1, car_trns)
    car_trns = np.where(scd.iloc[1:,8]=='Lost National Semifinal', car_trns.astype(float)-1, car_trns)
    car_trns = np.where(scd.iloc[1:,8]=='Lost National Final', car_trns.astype(float)-1, car_trns)
    car_trns = np.where(scd.iloc[1:,8]=='Won National Final', car_trns.astype(float)-1, car_trns)

    car_s16s = np.where(scd.iloc[1:,8]=='Lost Third Round', scd.iloc[1:,21].astype(float)-1, scd.iloc[1:,21])
    car_s16s = np.where(scd.iloc[1:,8]=='Lost Regional Semifinal', car_s16s.astype(float)-1, car_s16s)
    car_s16s = np.where(scd.iloc[1:,8]=='Lost Regional Final', car_s16s.astype(float)-1, car_s16s)
    car_s16s = np.where(scd.iloc[1:,8]=='Lost National Semifinal', car_s16s.astype(float)-1, car_s16s)
    car_s16s = np.where(scd.iloc[1:,8]=='Lost National Final', car_s16s.astype(float)-1, car_s16s)
    car_s16s = np.where(scd.iloc[1:,8]=='Won National Final', car_s16s.astype(float)-1, car_s16s)

    car_ffs = np.where(scd.iloc[1:,8]=='Lost National Semifinal', scd.iloc[1:,22].astype(float)-1, scd.iloc[1:,22])
    car_ffs = np.where(scd.iloc[1:,8]=='Lost National Final', car_ffs.astype(float)-1, car_ffs)
    car_ffs = np.where(scd.iloc[1:,8]=='Won National Final', car_ffs.astype(float)-1, car_ffs)

    car_chmps = np.where(scd.iloc[1:,8]=='Won National Final', scd.iloc[1:,23].astype(float)-1, scd.iloc[1:,23])

    np_arrays = [cas_wins,cas_losses, car_wins, car_losses, cas_trns, cas_s16s, cas_ffs, cas_chmps, car_trns, car_s16s, car_ffs, car_chmps]
    scd = pd.concat([scd.iloc[1:,0],scd.iloc[1:,1],scd.iloc[1:,2]],axis=1)
    for npa in np_arrays:
        nc = pd.Series(npa)
        scd = pd.concat([scd,nc],axis=1)

    # get rid of negatives
    # some coaches have more season games than career games, odd
    num = scd._get_numeric_data()
    num[num < 0] = 0

    scd = scd.iloc[1:,:]
    scd.columns=['Coach','School','Conf', 'SchoolWins','SchoolLosses','CareerWins','CareerLosses',
    'STourns','SS16s','SFFs','SChmps','CTourns','CS16s','CFFs','CChmps']
    scd['Season'] = season
    return scd

# load all data you can
def add_everything(df):
    tc = pd.read_csv('../data/Stage2/TeamConferences.csv')
    df = pd.merge(left=df,right=tc,left_on=['Season', 'WTeamID'],right_on=['Season','TeamID'])
    df = pd.merge(left=df,right=tc,left_on=['Season', 'LTeamID'],right_on=['Season','TeamID'],
    suffixes=['W','L'])
    df = df.drop(columns=['TeamIDW','TeamIDL'])

    ts = pd.read_csv('../data/Stage2/TeamSpellings.csv',encoding="ISO-8859-1")

    seasons = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
    cd = None
    for season in seasons:
        # season coaching data
        scd = pd.read_csv('../data/coaches/'+str(season)+'.csv')
        scd = fix_scd(scd,season)
        scd['School'] = scd['School'].str.lower()
        scd = pd.merge(left=scd,right=ts,left_on=['School'],right_on=['TeamNameSpelling'])
        if cd is None:
            cd = scd
        else:
            cd = pd.concat([cd,scd])

    cd = cd.drop(columns=['TeamNameSpelling','Conf'])
    cd = cd.sort_values(by='Season')
    print(len(df))
    df = pd.merge(left=df,right=cd,left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
    df = pd.merge(left=df,right=cd,left_on=['Season','LTeamID'],right_on=['Season','TeamID'],suffixes=['W','L'])
    df = df.drop_duplicates()
    df = df.drop(columns=['TeamIDL','TeamIDW'])

    #merge vegas
    # vegas = pd.read_csv('../data/Vegas/master.csv')
    # print(len(vegas))
    # print(len(df))
    # print(df.head())
    # wvegas = vegas[['Season','DayNum','WTeamID','W_line']]
    # lvegas = vegas[['Season','DayNum','LTeamID','L_line']]
    # df = pd.merge(left=df,right=wvegas,left_on=['Season','DayNum','WTeamID'],
    # right_on=['Season', 'DayNum', 'WTeamID'])
    # df = pd.merge(left=df,right=lvegas,left_on=['Season','DayNum','LTeamID'],
    # right_on=['Season', 'DayNum', 'LTeamID'],suffixes=['W','L'])
    # print(len(df))
    # print(df.head())
    # raise ValueError
    return df
sdf = pd.read_csv('../data/Stage2/RegularSeasonDetailedResults.csv')
tdf = pd.read_csv('../data/Stage2/NCAATourneyDetailedResults.csv')

sdf = add_everything(sdf)
tdf = add_everything(tdf)

def one_per_line(df):
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
        if col[:1]=='W' and col != 'WLoc':
            wdf = wdf.rename(columns={
                col:col[1:]
            })
            ldf = ldf.rename(columns={
                col:'Opp'+col[1:]
            })
        if col[:1]=='L' and col != 'WLoc':
            wdf = wdf.rename(columns={
                col:'Opp'+col[1:]
            })
            ldf = ldf.rename(columns={
                col:col[1:]
            })
    for col in list(wdf):
        if col[-1:]=='W':
            wdf = wdf.rename(columns={
                col:col[:-1]
            })
            ldf = ldf.rename(columns={
                col:'Opp'+col[:-1]
            })
        if col[-1]=='L':
            wdf = wdf.rename(columns={
                col:'Opp'+col[:-1]
            })
            ldf = ldf.rename(columns={
                col:col[:-1]
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
    return df

print(len(sdf))
sdf = one_per_line(sdf)
sdf = sdf.drop_duplicates()
tdf = one_per_line(tdf)
print(len(sdf))
elo_ratings = pd.read_csv('../data/elo_ratings.csv')
elo_ratings = elo_ratings.drop(columns=['Relo'])
sdf = pd.merge(left=sdf,right=elo_ratings,on=['Season', 'DayNum', 'TeamID'])
elo_ratings = elo_ratings.rename(columns={
    'TeamID':'OppTeamID',
    'Elo':'OppElo',
})
sdf = pd.merge(left=sdf,right=elo_ratings,on=['Season', 'DayNum', 'OppTeamID'])

sdf = sdf.drop(columns=['Unnamed: 0_x','Unnamed: 0_y'])

sdf.to_csv('./sdf_raw.csv',index=False)
tdf.to_csv('./tdf_raw.csv',index=False)
