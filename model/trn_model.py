import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting (when it can't be done under the pandas hood)

# Input data files are available in the "../input/" directory.
import os
data_directory = '../data/'

df_seeds = pd.read_csv(os.path.join(data_directory,'NCAATourneySeeds.csv'))
seeds19 = pd.read_csv(os.path.join(data_directory,'Stage2/NCAATourneySeeds.csv'))

df_seeds = pd.concat([df_seeds,seeds19])

df_sub = df_seeds.merge(df_seeds, how='inner', on='Season')
df_sub = df_sub[df_sub['TeamID_x'] < df_sub['TeamID_y']]

df_sub['ID'] = df_sub['Season'].astype(str) + '_' \
              + df_sub['TeamID_x'].astype(str) + '_' \
              + df_sub['TeamID_y'].astype(str)

df_sub['SeedInt_x'] = [int(x[1:3]) for x in df_sub['Seed_x']]
df_sub['SeedInt_y'] = [int(x[1:3]) for x in df_sub['Seed_y']]

df_sub['SeedDiff'] = df_sub['SeedInt_y'] - df_sub['SeedInt_x']

df_sub = df_sub.rename(columns={
    'TeamID_x':'Team1',
    'TeamID_y':'Team2'
})
df_sub = df_sub[['Season','Team1','Team2','SeedDiff']]

df_sub.to_csv('../massey/seeds.csv',index=False)

df_sub.loc[(df_sub['Season'] >= 2014) & (df_sub['Season'] <= 2018), ['ID', 'Pred']].to_csv('./Submission.csv',index=False)

# now pare down existing df_sub
df_sub = df_sub[['ID','Pred']]
print(df_sub.head())



#end
