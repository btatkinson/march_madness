import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

minput = pd.read_csv('./input.csv')

X_cols = ['DayNum', 'Loc', 'SeaORtg', 'SeaDRtg', 'SeaOSR', 'SeaDSR', 'Elo', 'OppSeaORtg', 'OppSeaDRtg', 'OppSeaOSR', 'OppSeaDSR', 'OppElo']
X = minput[X_cols].values
y = minput.Result.values

print("You are using "+ str(len(X_cols)) + " features")

# print(minput[['Adj_SeaORtg', 'Adj_SeaDRtg', 'Adj_SeaOSR', 'Adj_SeaDSR']].describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
gbm = lgb.LGBMRegressor(num_leaves=7,
                        learning_rate=0.09,
                        n_estimators=100)

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=10)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

print('The log loss of prediction is:', log_loss(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importances_,X_cols)), columns=['Value','Feature'])
feature_imp = feature_imp.sort_values(by="Value", ascending=False)
feature_imp.to_csv('feature_importances.csv', index=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()

# get last rating for each team
#
sub_df = pd.read_csv('../data/SampleSubmissionStage1.csv')

print(len(sub_df))

sub_df['Season'] = sub_df['ID'].map(lambda x: int(x.split('_')[0]))
sub_df['Team1'] = sub_df['ID'].map(lambda x: int(x.split('_')[1]))
sub_df['Team2'] = sub_df['ID'].map(lambda x: int(x.split('_')[2]))

# historical tournament benchmark
trn_input = pd.read_csv('./trn_input.csv')

sub_df = sub_df.merge(trn_input[['Season','TeamID','SeaORtg', 'SeaDRtg', 'SeaOSR', 'SeaDSR','Elo']],how='left',
                                    left_on = ['Season','Team1'], right_on = ['Season','TeamID'])
sub_df = sub_df.merge(trn_input[['Season','TeamID','SeaORtg', 'SeaDRtg', 'SeaOSR', 'SeaDSR','Elo']],how='left',
                                    left_on = ['Season','Team2'], right_on = ['Season','TeamID'],
                                   suffixes=['','L'])
sub_df['Loc'] = 0.5
sub_df['DayNum'] = 133
sub_df = sub_df.drop(columns=['TeamIDL'])

sub_df = sub_df.rename(columns={
    'SeaORtgL':'OppSeaORtg',
    'SeaDRtgL':'OppSeaDRtg',
    'SeaOSRL':'OppSeaOSR',
    'SeaDSRL':'OppSeaDSR',
    'EloL':'OppElo'
})
sub_df['Target'] = 1
y_true = sub_df.Target.values
# y_true[0] = 0
input_df = sub_df[X_cols].values
sub_df['Pred'] = gbm.predict(input_df, num_iteration=gbm.best_iteration_)
print(sub_df.Pred.values)
y_pred=sub_df.Pred.values
print("Tournament log loss is " + str(log_loss(y_true, y_pred, labels=[0,1])))

sub_df[['ID', 'Pred']].to_csv('submission.csv', index=False)
# print(sub_df[['ID', 'Pred']].head())




# end
