
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re, random, time
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_colwidth = 1000
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV


# In[4]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[7]:


sample = pd.read_csv("sample_submission.csv")


# In[12]:


train_size = train.groupby(['matchId','groupId']).size().reset_index(name='group_size')
test_size = test.groupby(['matchId','groupId']).size().reset_index(name='group_size')

train_mean = train.groupby(['matchId','groupId']).mean().reset_index()
test_mean = test.groupby(['matchId','groupId']).mean().reset_index()

train_max = train.groupby(['matchId','groupId']).max().reset_index()
test_max = test.groupby(['matchId','groupId']).max().reset_index()

train_min = train.groupby(['matchId','groupId']).min().reset_index()
test_min = test.groupby(['matchId','groupId']).min().reset_index()


# In[13]:


train_match_mean = train.groupby(['matchId']).mean().reset_index()
test_match_mean = test.groupby(['matchId']).mean().reset_index()

train = pd.merge(train, train_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
test = pd.merge(test, test_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
del train_mean
del test_mean

train = pd.merge(train, train_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
test = pd.merge(test, test_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
del train_max
del test_max

train = pd.merge(train, train_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
test = pd.merge(test, test_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
del train_min
del test_min

train = pd.merge(train, train_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
test = pd.merge(test, test_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
del train_match_mean
del test_match_mean

train = pd.merge(train, train_size, how='left', on=['matchId', 'groupId'])
test = pd.merge(test, test_size, how='left', on=['matchId', 'groupId'])
del train_size
del test_size

train_columns = list(test.columns)


# In[14]:


train_columns.remove("Id")
train_columns.remove("matchId")
train_columns.remove("groupId")
train_columns.remove("Id_mean")
train_columns.remove("Id_max")
train_columns.remove("Id_min")
train_columns.remove("Id_match_mean")


# In[15]:


train_columns_new = []
for name in train_columns:
    if '_' in name:
        train_columns_new.append(name)
train_columns = train_columns_new    


# In[16]:


X_train = train[train_columns]
X_test = test[train_columns]


# Feature engineering

# In[18]:


df_train = X_train


# In[19]:


X = df_train
y = train.winPlacePerc


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# np.random.seed(1)
# train_rows = np.random.choice(df_train.index, int(len(df_train)* 0.8), replace=False)
# valid_rows = [x for x in df_train.index if x not in train_rows]
# df_train1 = df_train.loc[train_rows]
# df_valid1 = df_train.loc[valid_rows]
# 
# train_X = df_train1.drop(['Id', 'groupId', 'matchId'], axis = 1)
# train_Y = df_train1.winPlacePerc
# valid_X = df_valid1.drop(['Id', 'groupId', 'matchId'], axis = 1)
# valid_Y = df_valid1.winPlacePerc

# train_X = df_train1.drop(['Id', 'groupId', 'matchId' , 'winPlacePerc'], axis = 1)
# 
# valid_X = df_valid1.drop(['Id', 'groupId', 'matchId' , 'winPlacePerc'], axis = 1)

# In[21]:


train_X = X_train
train_Y = y_train
valid_X = X_test
valid_Y = y_test


# In[24]:


dtrain = xgb.DMatrix(train_X, label=train_Y.values)
dtest = xgb.DMatrix(valid_X , label  = valid_Y.values)
evallist  = [(dtest,'eval'), (dtrain,'train')]


# In[27]:


params = {
    'eta': 0.5, 
    'boosting': 'gbtree', 
    'objective': 'reg:logistic', 
    'eval_metric': 'mae', 
    'is_training_metric': False, 
    'scale_pos_weight': 0.5,
    'max_depth': 15,  
    'min_child_samples': 100,  
    'max_bin': 100,  
    'subsample': 0.7,  
    'subsample_freq': 1,  
    'colsample_bytree': 0.7,
    'seed': 0
}


# In[213]:


fit_model = xgb.train( params, dtrain , num_boost_round = 200, evals=   evallist , early_stopping_rounds= 10 )
print('Plot feature importances...')
ax = xgb.plot_importance(fit_model)
plt.show()


# In[31]:


gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(9,12)
    for min_child_weight in range(5,8)
]


# In[38]:


best_params = None
num_boost_round = 999
params = {}
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=2,
        metrics={'mae'},
        early_stopping_rounds=10
    )


# model = xgb.XGBClassifier(objective = 'reg:logistic')
# param_dist = {"max_depth": [10,30,50],
#               "min_child_weight" : [1,3,6],
#               "n_estimators": [200],
#               "learning_rate": [0.05, 0.1,0.16]}
# grid_search = RandomizedSearchCV(model , param_distributions= param_dist, cv = 2)

# grid_search.fit(X_train, y_train)

# In[36]:


X_test = test[train_columns]


# In[37]:


test_X = X_test
dtest_X = xgb.DMatrix(test_X)


# In[216]:


output_win = fit_model.predict(dtest_X)


# In[218]:


output_df = pd.DataFrame( output_win)


# In[39]:


sampleID = sample['Id']


# In[220]:


t = pd.concat([sampleID , output_df] , axis = 1)


# In[221]:


t.columns = ['Id' , 'winPLacePerc']


# In[222]:


t.to_csv('Submission_10.csv' , index = False)

