# coding: utf-8
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Imputer
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math

# Set the path you store data
path = '<Your root path>'

# According to the num_games, assign the rank for each user.
tr9 = pd.read_csv(path+'train9.csv')
te9 = pd.read_csv(path+'test9.csv')
tr1 = pd.read_csv(path+'train1.csv')
te1 = pd.read_csv(path+'test1.csv')
tmp = pd.concat([tr9, tr1, te9, te1],axis=0)
tmp = tmp.sort_values(['user_id','num_games'],ascending=[1,0])
def assign_times(dataframe):
    dataframe = dataframe.reset_index(drop=True)
    dataframe['index'] = dataframe.index + 1
    dataframe['times'] = dataframe['index'] % 10
    dataframe = dataframe.drop(['index'],axis = 1)
    return dataframe
tmp = assign_times(tmp)
tmp.loc[tmp.times ==0,'times'] = 10

# Merge with hero data
hero = pd.read_csv(path+'hero_data.csv')
df = pd.merge(tmp, hero, how='left', on='hero_id')

# Check user_id in train and test data
# There is no overlap between train and test users
tr_user = np.unique(tr1.user_id)
te_user = np.unique(te1.user_id)
print(len(tr_user))
print(len(te_user))
print(len(set(tr_user)-set(te_user)))

# There are totally 115 heroes show up in train/test data
# Train data include all possible heroes
tr1_herotype = np.unique(tr1.hero_id)
tr9_herotype = np.unique(tr9.hero_id)
te1_herotype = np.unique(te1.hero_id)
te9_herotype = np.unique(te9.hero_id)
df_herotype = np.unique(df.hero_id)
print(len(te1_herotype))
print(len(df_herotype))
print(set(te1_herotype) - set(df_herotype))

# Return columns' name with missing value
print(df.isnull().any()[df.isnull().any() ==True])

## Feature engineering for heroes
# Heroes' explosure(how many times they are chosen)
hero_showup = df.groupby('hero_id').agg({'num_games':'sum'})
hero_showup.columns = ['total_num_games']
hero_showup['hero_id'] = hero_showup.index

# Calculate Heroes' num_wins, mean win_ratio and mean kda_ratio
tmp = df.loc[~df.kda_ratio.isnull()]
hero_feature = tmp.groupby('hero_id').agg({'num_games':'sum','num_wins':'sum','kda_ratio':'mean'})
hero_feature['win_ratio'] = hero_feature.num_wins / hero_feature.num_games
hero_feature = hero_feature.drop(['num_games'],axis=1)
hero_feature.columns = ['total_num_wins','mean_kda_ratio','mean_win_retio']
hero_feature['hero_id'] = hero_feature.index

df = pd.merge(df, hero_showup, how='left', on='hero_id')
df = pd.merge(df, hero_feature, how='left', on='hero_id')


## Feature engineering for users
# How many games the user play
user_showup = df.groupby('user_id').agg({'num_games':'sum'})
user_showup.columns = ['total_user_num_games']
user_showup['user_id'] = user_showup.index

# Calculate users' num_wins, mean win_ratio and mean kda_ratio
tmp = df.loc[~df.kda_ratio.isnull()]
user_feature = tmp.groupby('user_id').agg({'num_games':'sum','num_wins':'sum','kda_ratio':'mean'})
user_feature['win_ratio'] = user_feature.num_wins / user_feature.num_games
user_feature = user_feature.drop(['num_games'],axis=1)
user_feature.columns = ['total_user_num_wins','mean_user_kda_ratio','mean_user_win_retio']
user_feature['user_id'] = user_feature.index

df = pd.merge(df, user_showup, how='left', on='user_id')
df = pd.merge(df, user_feature, how='left', on='user_id')


## Users' performance on each primary_attr
# num_games on each primary_attr
user_primary = df.groupby(['user_id','primary_attr'],as_index=False).agg({'num_games':'sum'})
user_primary.columns = ['user_id','primary_attr','total_user_primary_num_games']
user_primary = user_primary.pivot_table(index='user_id',
                          columns='primary_attr',values='total_user_primary_num_games').reset_index()
user_primary.columns = ['user_id','total_num_games_agi','total_num_games_int','total_num_games_str']


# mean win_ratio, num_wins and kda_ratio
tmp = df.loc[~df.kda_ratio.isnull()]
user_primary_feature = tmp.groupby(['user_id','primary_attr'],as_index=False).agg({'num_games':'sum',
                                                                        'num_wins':'sum','kda_ratio':'mean'})
user_primary_feature['win_ratio'] = user_primary_feature.num_wins / user_primary_feature.num_games
user_primary_feature = user_primary_feature.drop(['num_games'],axis=1)
user_primary_feature.columns = ['user_id','primary_attr',
                                'total_user_primary_num_wins','mean_user_primary_kda_ratio',
                                'mean_user_primary_win_ratio']
user_primary_feature = user_primary_feature.pivot_table(index='user_id',
                          columns='primary_attr',values=['total_user_primary_num_wins',
                                                         'mean_user_primary_kda_ratio',
                                                        'mean_user_primary_win_ratio']).reset_index()
user_primary_feature.columns = ['user_id',
    'mean_user_primary_kda_ratio_agi','mean_user_primary_kda_ratio_int','mean_user_primary_kda_ratio_str',
    'mean_user_primary_win_retio_agi','mean_user_primary_win_retio_int','mean_user_primary_win_retio_str',
    'total_user_primary_num_wins_agi','total_user_primary_num_wins_int','total_user_primary_num_wins_str']

df = pd.merge(df, user_primary, how='left', on='user_id')
df = pd.merge(df, user_primary_feature, how='left', on='user_id')



## Transform categorical variables to numeric
# primary_attr
df = pd.concat([df, pd.get_dummies(df.primary_attr,prefix='primary_attr')], axis=1)
df = df.drop(['primary_attr'],axis=1)

# attack_type
le = LabelEncoder()
df.attack_type = le.fit_transform(df.attack_type)

# roles
flat_list = [item for sublist in df.roles.str.split(':') for item in sublist]
role_list = [element for element in set(flat_list)]
df['roles_seq'] = df.roles.str.split(':')
role_match_variable = pd.DataFrame(df['roles_seq'].apply(lambda x: [int(i in x) for i in role_list]).tolist(),
                                  columns=[name for name in role_list])
df = pd.concat([df, role_match_variable],axis=1)

# The number of roles the hero is
num_roles = df[role_list].sum(axis=1)
df['num_roles'] = num_roles

# Convert roles' dummy variable to weight numeric variables
for i in role_list:
    df['{}_V'.format(i)] = df[i] / df['num_roles']

##  Clustering for heroes' roles
test = df.loc[df.kda_ratio.isnull(),:]
df = df.loc[~df.kda_ratio.isnull(),:]
df = df.reset_index(drop=True)
test = test.reset_index(drop=True)

# The number of cluster you want
n_cluster = 3

# Fit and transform for training data
kmeans = MiniBatchKMeans(n_clusters=n_cluster, batch_size=1000).fit(df[['Nuker_V', 'Disabler_V',
       'Pusher_V', 'Durable_V', 'Jungler_V', 'Initiator_V', 'Escape_V',
       'Carry_V', 'Support_V']].values)

role_group = kmeans.predict(df[['Nuker_V', 'Disabler_V',
       'Pusher_V', 'Durable_V', 'Jungler_V', 'Initiator_V', 'Escape_V',
       'Carry_V', 'Support_V']].values)
rg = pd.DataFrame({'role_group':role_group})
df = pd.concat([df, rg], axis=1)

# transform for testing data
role_group = kmeans.predict(test[['Nuker_V', 'Disabler_V',
       'Pusher_V', 'Durable_V', 'Jungler_V', 'Initiator_V', 'Escape_V',
       'Carry_V', 'Support_V']])
rg = pd.DataFrame({'role_group':role_group})
test = pd.concat([test, rg], axis=1)

# to dummy variables
df = pd.concat([df, test], axis=0)
df = df.reset_index(drop=True)
df = pd.concat([df, pd.get_dummies(df.role_group,prefix='role_group')], axis=1)


## Clustering for Heroes
test = df.loc[df.kda_ratio.isnull(),:]
df = df.loc[~df.kda_ratio.isnull(),:]
df = df.reset_index(drop=True)
test = test.reset_index(drop=True)
print(len(df))
print(len(test))

# The number of cluster you want
n_cluster = 5

# Fit and transform for training data
kmeans = MiniBatchKMeans(n_clusters=n_cluster, batch_size=1000).fit(df[['base_health',
       'base_health_regen', 'base_mana', 'base_mana_regen', 'base_armor',
       'base_magic_resistance', 'base_attack_min', 'base_attack_max',
       'base_strength', 'base_agility', 'base_intelligence',
       'strength_gain', 'agility_gain', 'intelligence_gain',
       'attack_range', 'projectile_speed', 'attack_rate', 'move_speed',
       'turn_rate']].values)

hero_group = kmeans.predict(df[['base_health',
       'base_health_regen', 'base_mana', 'base_mana_regen', 'base_armor',
       'base_magic_resistance', 'base_attack_min', 'base_attack_max',
       'base_strength', 'base_agility', 'base_intelligence',
       'strength_gain', 'agility_gain', 'intelligence_gain',
       'attack_range', 'projectile_speed', 'attack_rate', 'move_speed',
       'turn_rate']].values)
hg = pd.DataFrame({'hero_group':hero_group})
df = pd.concat([df, hg], axis=1)

# Transform for testing data
hero_group = kmeans.predict(test[['base_health',
       'base_health_regen', 'base_mana', 'base_mana_regen', 'base_armor',
       'base_magic_resistance', 'base_attack_min', 'base_attack_max',
       'base_strength', 'base_agility', 'base_intelligence',
       'strength_gain', 'agility_gain', 'intelligence_gain',
       'attack_range', 'projectile_speed', 'attack_rate', 'move_speed',
       'turn_rate']])
hg = pd.DataFrame({'hero_group':hero_group})
test = pd.concat([test, hg], axis=1)

# To dummy variables
df = pd.concat([df, test], axis=0)
df = df.reset_index(drop=True)
df = pd.concat([df, pd.get_dummies(df.hero_group,prefix='group')], axis=1)

# How many heroes in each cluster
print(df.groupby('hero_group').agg({'hero_id':'nunique'}))


# Features for each hero cluster
# num_games
hero_group_showup = df.groupby('hero_group').agg({'num_games':'sum'})
hero_group_showup.columns = ['total_hero_group_num_games']
hero_group_showup['hero_group'] = hero_group_showup.index

tmp = df.loc[~df.kda_ratio.isnull()]
# mean win_ratio, num_wins and kda_ratio
hero_group_feature = tmp.groupby('hero_id').agg({'num_games':'sum','num_wins':'sum','kda_ratio':'mean'})
hero_group_feature['win_ratio'] = hero_group_feature.num_wins / hero_group_feature.num_games
hero_group_feature = hero_group_feature.drop(['num_games'],axis=1)
hero_group_feature.columns = ['total_hero_group_num_wins','mean_hero_group_kda_ratio','mean_hero_group_win_retio']
hero_group_feature['hero_group'] = hero_group_feature.index


df = pd.merge(df, hero_group_showup, how='left', on='hero_group')
df = pd.merge(df, hero_group_feature, how='left', on='hero_group')



## Users' performance on each hero cluster
# num_games
tmp = df.groupby(['user_id','hero_group'],as_index=False).agg({'num_games':'sum'})
tmp = tmp.pivot_table(index='user_id',
                          columns='hero_group',values='num_games').reset_index()
tmp = tmp.fillna(0)
tmp.columns = ['user_id','num_hg0','num_hg1','num_hg2','num_hg3','num_hg4']
df = pd.merge(df, tmp, how='left', on='user_id')

# num_wins, mean win_ratio and kda_ratio
tmp = df.loc[~df.kda_ratio.isnull()]
tmp = tmp.groupby(['user_id','hero_group'],as_index=False).agg({'kda_ratio':'mean','num_wins':'sum',
                                                                'num_games':'sum'})
tmp['win_ratio'] = tmp['num_wins'] / tmp['num_games']
tmp = tmp.pivot_table(index='user_id',
                          columns='hero_group',values=['num_games','num_wins','kda_ratio','win_ratio']).reset_index()
tmp.columns =['user_id','kda_ratio_g0','kda_ratio_g1','kda_ratio_g2','kda_ratio_g3','kda_ratio_g4',
             'num_games_g0','num_games_g1','num_games_g2','num_games_g3','num_games_g4',
             'num_wins_g0','num_wins_g1','num_wins_g2','num_wins_g3','num_wins_g4',
             'win_ratio_g0','win_ratio_g1','win_ratio_g2','win_ratio_g3','win_ratio_g4'
             ]
tmp = tmp.fillna(0)
tmp = tmp.drop(['num_games_g0','num_games_g1','num_games_g2','num_games_g3','num_games_g4'], axis=1)
df = pd.merge(df, tmp, how='left', on='user_id')

## Save final data if you want
#df.to_csv(path+'df.csv',index=False)

var = [ 'num_games',
       'times', 'attack_type','base_health', 'base_health_regen',
       'base_mana', 'base_mana_regen', 'base_armor',
       'base_magic_resistance', 'base_attack_min', 'base_attack_max',
       'base_strength', 'base_agility', 'base_intelligence',
       'strength_gain', 'agility_gain', 'intelligence_gain',
       'attack_range', 'projectile_speed', 'attack_rate', 'move_speed',
       'turn_rate', 'total_num_games', 'total_num_wins', 'mean_kda_ratio',
       'mean_win_retio', 'total_user_num_games', 'total_user_num_wins',
       'mean_user_kda_ratio', 'mean_user_win_retio',
       'primary_attr_agi',
       'primary_attr_int', 'primary_attr_str','Nuker',
       'Disabler', 'Pusher', 'Durable', 'Jungler', 'Initiator', 'Escape',
       'Carry', 'Support','group_0','group_1',
       'group_2', 'group_3', 'group_4']


# train-test split
train = df.loc[~df.kda_ratio.isnull()]
test = df.loc[df.kda_ratio.isnull()]
train_X = train[var]
train_y = train['kda_ratio']
train_y2 = np.log(train['kda_ratio']+1)
test_X = test[var]

## grid search
#model1 = xgb.XGBRegressor()
#parameters = {'n_estimators':[700,800,900], 'max_depth':range(8,10)}
#model1 = GridSearchCV(model, parameters,cv=3)
#model1.fit(train_X, train_y2)

model = xgb.XGBRegressor(n_estimators=800, max_depth=9,seed=2018)
model.fit(train_X, train_y2)

pre = model.predict(test_X)
submit = pd.DataFrame({'id':test.id,'kda_ratio':np.exp(pre)-1})
submit.to_csv(path + 'submit.csv')

