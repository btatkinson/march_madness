import pandas as pd
import numpy as np


df_sub = pd.read_csv('submission.csv')

df_sub['Season'] = df_sub['ID'].map(lambda x: int(x.split('_')[0]))
df_sub['Team1'] = df_sub['ID'].map(lambda x: int(x.split('_')[1]))
df_sub['Team2'] = df_sub['ID'].map(lambda x: int(x.split('_')[2]))

seeds = pd.read_csv('../data/Stage2/NCAATourneySeeds.csv')

df_sub = pd.merge(left=df_sub,right=seeds,left_on=['Season','Team1'],right_on=['Season','TeamID'])
df_sub = pd.merge(left=df_sub,right=seeds,left_on=['Season','Team2'],right_on=['Season','TeamID'],suffixes=['_1','_2'])

df_sub = df_sub.drop(columns=['TeamID_1','TeamID_2'])

df_sub['s1_region'] = df_sub['Seed_1'].str[:1]
df_sub['s2_region'] = df_sub['Seed_2'].str[:1]

df_sub['is_chmp'] = np.where(((df_sub['s1_region']=='W') & (df_sub['s2_region']=='Y')), 1, 0)
df_sub['is_chmp'] = np.where(((df_sub['s1_region']=='W') & (df_sub['s2_region']=='Z')), 1, df_sub['is_chmp'])
df_sub['is_chmp'] = np.where(((df_sub['s1_region']=='X') & (df_sub['s2_region']=='Y')), 1, df_sub['is_chmp'])
df_sub['is_chmp'] = np.where(((df_sub['s1_region']=='X') & (df_sub['s2_region']=='Z')), 1, df_sub['is_chmp'])
df_sub['is_chmp'] = np.where(((df_sub['s1_region']=='Y') & (df_sub['s2_region']=='W')), 1, df_sub['is_chmp'])
df_sub['is_chmp'] = np.where(((df_sub['s1_region']=='Y') & (df_sub['s2_region']=='X')), 1, df_sub['is_chmp'])
df_sub['is_chmp'] = np.where(((df_sub['s1_region']=='Z') & (df_sub['s2_region']=='W')), 1, df_sub['is_chmp'])
df_sub['is_chmp'] = np.where(((df_sub['s1_region']=='Z') & (df_sub['s2_region']=='X')), 1, df_sub['is_chmp'])

df_sub['Pred'][df_sub['is_chmp']  == 1] = 1

df_sub['ID'] = df_sub['Season'].astype(str) + '_' \
              + df_sub['Team1'].astype(str) + '_' \
              + df_sub['Team2'].astype(str)

df_sub['Pred'][df_sub['Pred'] > .965] = 1
df_sub['Pred'][df_sub['Pred'] < .045] = 0

df_sub = df_sub[['ID','Pred']]
print(df_sub)
df_sub.to_csv('sub2.csv',index=False)

# regions = pd.read_csv('../data/NCAATourneySlots.csv')

# df_sub = pd.merge(left=df_sub,right=regions,left_on=['Season','Seed_1','Seed_2'],right_on=['Season','StrongSeed','WeakSeed'])


df_tc = pd.read_csv('../data/NCAATourneyCompactResults.csv')

df_tc['ID'] = df_tc['Season'].astype(str) + '_' \
              + (np.minimum(df_tc['WTeamID'],df_tc['LTeamID'])).astype(str) + '_' \
              + (np.maximum(df_tc['WTeamID'],df_tc['LTeamID'])).astype(str)

df_tc['Result'] = 1*(df_tc['WTeamID'] < df_tc['LTeamID'])





def kaggle_clip_log(x):
    '''
    Calculates the natural logarithm, but with the argument clipped within [1e-15, 1 - 1e-15]
    '''
    return np.log(np.clip(x,1.0e-15, 1.0 - 1.0e-15))

def kaggle_log_loss(pred, result):
    '''
    Calculates the kaggle log loss for prediction pred given result result
    '''
    return -(result*kaggle_clip_log(pred) + (1-result)*kaggle_clip_log(1.0 - pred))

def score_submission(df_sub, df_results, on_season=None, return_df_analysis=True):
    '''
    Scores a submission against relevant tournament results

    Parameters
    ==========
    df_sub: Pandas dataframe containing predictions to be scored (must contain a column called 'ID' and
            a column called 'Pred')

    df_results: Pandas dataframe containing results to be compared against (must contain a column
            called 'ID' and a column called 'Result')

    on_season: array-like or None.  If array, should contain the seasons for which a score should
            be calculated.  If None, will use all seasons present in df_results

    return_df_analysis: Bool.  If True, will return the dataframe used for calculations.  This is useful
            for future analysis

    Returns
    =======
    df_score: pandas dataframe containing the average score over predictions that were scorable per season
           as well as the number of obvious errors encountered
    df_analysis:  pandas dataframe containing information about all results used in scoring
                  Only provided if return_df_analysis=True
    '''

    df_analysis = df_results.copy()

    # this will overwrite if there's already a season column but it should be the same
    df_analysis['Season'] = [int(x.split('_')[0]) for x in df_results['ID']]

    if not on_season is None:
        df_analysis = df_analysis[np.in1d(df_analysis['Season'], on_season)]

    # left merge with the submission.  This will keep all games for which there
    # are results regardless of whether there is a prediction
    df_analysis = df_analysis.merge(df_sub, how='left', on='ID')

    # check to see if there are obvious errors in the predictions:
    # Obvious errors include predictions that are less than 0, greater than 1, or nan
    # You can add more if you like
    df_analysis['ObviousError'] = 1*((df_analysis['Pred'] < 0.0) \
                                  | (df_analysis['Pred'] > 1.0) \
                                  | (df_analysis['Pred'].isnull()))

    df_analysis['LogLoss'] = kaggle_log_loss(df_analysis['Pred'], df_analysis['Result'])

    df_score = df_analysis.groupby('Season').agg({'LogLoss' : 'mean', 'ObviousError': 'sum'})

    if return_df_analysis:
        return df_score, df_analysis
    else:
        return df_score

print(df_sub.head())

# print(df_sub)
df_score, df_analysis = score_submission(df_sub, df_tc, on_season = np.arange(2014,2019), return_df_analysis=True)
print(df_score)

# end
