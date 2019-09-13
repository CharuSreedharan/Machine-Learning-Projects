#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime

import os
print(os.listdir("../input"))


# In[ ]:


from sklearn import *
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# In[ ]:


(market_train, news_train) = env.get_training_data()


# In[ ]:


# Preprocessing steps

# Deleting data prior 2009
market_train = market_train.loc[market_train['time'].dt.date>=datetime.date(2009,1,1)]
news_train = news_train.loc[news_train['time'].dt.date>=datetime.date(2009,1,1)]

# Clipping excess close-open difference
market_train['c_open_ratio'] = np.abs(market_train['close']/market_train['open'])
threshold = 0.5
print('In %i lines price increases by 50%% or more in a day' %(market_train['c_open_ratio']>=1.5).sum())
print('In %i lines price decreases by 50%% or more in a day' %(market_train['c_open_ratio']<=0.5).sum())


market_train = market_train.loc[market_train['c_open_ratio'] < 1.5]
market_train = market_train.loc[market_train['c_open_ratio'] > 0.5]
market_train = market_train.drop(columns=['c_open_ratio'])

# Filling mktres with raw values
column_marketres = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
column_raw = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10']
for i in range(len(column_raw)):
    market_train[column_marketres[i]] = market_train[column_marketres[i]].fillna(market_train[column_raw[i]])
    
# Clipping excess returns
column_return = column_marketres + column_raw + ['returnsOpenNextMktres10']
orig_len = market_train.shape[0]
for column in column_return:
    market_train = market_train.loc[market_train[column]>=-2]
    market_train = market_train.loc[market_train[column]<=2]
new_len = market_train.shape[0]
rmv_len = np.abs(orig_len-new_len)
print('There were %i lines removed' %rmv_len)


# Removing unknown assetname
orig_len = market_train.shape[0]
market_train = market_train[~market_train['assetCode'].isin(['EBRYY.OB'])]
new_len = market_train.shape[0]
rmv_len = np.abs(orig_len-new_len)
print('There were %i lines removed' %rmv_len)


# Function to remove outliers
def remove_outliers(data_frame, column_list, low=0.02, high=0.98):
    for column in column_list:
        this_column = data_frame[column]
        quant_df = this_column.quantile([low,high])
        low_limit = quant_df[low]
        high_limit = quant_df[high]
        data_frame[column] = data_frame[column].clip(lower=low_limit, upper=high_limit)
    return data_frame



# Function to Remove outliers in news
columns_outlier = ['takeSequence', 'bodySize', 'sentenceCount', 'wordCount', 'sentimentWordCount', 'firstMentionSentence','noveltyCount12H',                  'noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H', 'volumeCounts24H',                  'volumeCounts3D','volumeCounts5D','volumeCounts7D']
news_train = remove_outliers(news_train, columns_outlier)


# In[ ]:


def preprocess_news(news_train):
    drop_list = [
        'audiences', 'subjects', 'assetName',
        'headline'
    ]
    news_train.drop(drop_list, axis=1, inplace=True)
    
    # Factorize categorical columns
    for col in ['headlineTag', 'provider', 'sourceId']:
        news_train[col], uniques = pd.factorize(news_train[col])
        del uniques
    
    # Remove {} and '' from assetCodes column
    news_train['assetCodes'] = news_train['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
    return news_train

news_train = preprocess_news(news_train)


# In[ ]:


import gc

def group_news(news_dataframe):
    news_dataframe['date'] = news_dataframe.time.dt.date 
    aggregations = ['mean']
    gp = news_dataframe.groupby(['assetCodes', 'date']).agg(aggregations)
    gp.columns = pd.Index(["{}_{}".format(e[0], e[1]) for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    # Converting to float
    float_cols = {c: 'float32' for c in gp.columns if c not in ['assetCodes', 'date']}
    return gp.astype(float_cols)

news_aggregate = group_news(news_train)
gc.collect()


# In[ ]:


# merge market and news dataframes
market_train['date'] = market_train.time.dt.date
df = market_train.merge(news_aggregate, how='left', left_on=['date', 'assetCode'], 
                            right_on=['date', 'assetCodes'])
del market_train, news_aggregate
gc.collect()
df.isna().sum()


# In[ ]:


# drop NA rows
df = df.dropna(axis=0)
df.shape


# In[ ]:


# initial target created
up = df.returnsOpenNextMktres10 >= 0


# In[ ]:


# columns not using
reqcol = [c for c in df if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                    'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                    'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe',
                                    'sourceTimestamp']]


# In[ ]:


X = df[reqcol].values
up = up.values
r = df.returnsOpenNextMktres10.values


# In[ ]:


# scaling
import numpy as np
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)


# In[ ]:


X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.25, random_state=0)


# In[ ]:


from xgboost import XGBClassifier
import time


# In[ ]:


xgb_model = XGBClassifier(n_jobs=4,n_estimators=250,max_depth=8,eta=0.1)


# In[ ]:


# Model fitting
xgb_model.fit(X_train,up_train)


# In[ ]:


xgb_model.score(X_test,up_test)


# In[ ]:


# Fetch Test set
dayss = env.get_prediction_days()


# In[ ]:


# Preprocessing the test set for submission
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
for (market_obs_df, news_obs_df, predictions_template_df) in (dayss):
    n_days +=1
    print(n_days,end=' ')
    t = time.time()
    news_obs_df = preprocess_news(news_obs_df)
    news_aggregate = group_news(news_obs_df)
    
    market_obs_df['date'] = market_obs_df.time.dt.date
    df = market_obs_df.merge(news_aggregate, how='left', left_on=['date', 'assetCode'], 
                            right_on=['date', 'assetCodes'])
    del market_obs_df, news_aggregate
    gc.collect()

    
    df = df[df.assetCode.isin(predictions_template_df.assetCode)]
    X_live = df[reqcol].values
    X_live = 1 - ((maxs - X_live) / rng)
    prep_time += time.time() - t
    
    t = time.time()
    lp = xgb_model.predict_proba(X_live)
    prediction_time += time.time() -t
    
    t = time.time()
    confidence = 2* lp[:,1] -1
    preds = pd.DataFrame({'assetCode':df['assetCode'],'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t


# In[ ]:


# submission
env.write_submission_file()

