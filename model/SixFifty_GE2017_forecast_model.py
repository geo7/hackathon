#!/usr/bin/env python
# coding: utf-8

# # SixFifty GE2017 Model
# ## Constituency Level Forecast Model
# For more information please see [SixFifty.org.uk](https://sixfifty.org.uk) or the [SixFifty Hackathon repo](https://github.com/six50/hackathon).

# In[1]:


# Libaries that may or may not be useful
import feather
import matplotlib
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn import ensemble, linear_model, metrics, model_selection, neural_network, tree

# Config
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
DATA_DIR = Path('../data/')
get_ipython().run_line_magic('matplotlib', 'inline')


# Running `data/retrieve_data.py` from the `hackathon` repo root will download the relevant datasets into the required location.

# ### Import model-ready General Election data for 2010 and 2015

# In[2]:


parties = ['con', 'lab', 'ld', 'ukip', 'grn']


# In[3]:


df = pd.read_feather(DATA_DIR / 'model' / 'ge_2010_2015_training_data.feather')
df.head(15)


# In[4]:


# Visualise feature relationships
sns.pairplot(df[['Electorate', 'Votes', 'votes_last', 'pc_last', 'polls_now', 'swing_now', 'swing_forecast_pc', 'actual_pc_now', 'party']],
             hue='party')


# In[5]:


# Some useful functions
def score_model(model, X, y, repeats=5):
    """Calculates a "5x5" (repeated 5-fold) cross-validated shuffled mean average error.
       Returns the mean across 5 (default) repeats."""
    mmae = []
    for i in range(repeats):
        mmae += [-np.mean(model_selection.cross_val_score(model, X=X, y=y, cv=model_selection.KFold(n_splits=5, shuffle=True), scoring='neg_mean_absolute_error'))]
    return np.mean(mmae)

def score_features(model, features):
    """Helper function to run scoring function for a feature subset and using percent voteshare.
       This equates to average error per party per seat."""
    return score_model(model=model, X=df[features], y=df['actual_pc_now'])


# ### UNS model
# Uniform National Swing has been pre-calculated here for you ("`swing_now`" and "`swing_forecast_pc`"). Let's evaluate how many seats are won by each party in 2015 according to this approach.

# In[6]:


df.head()


# In[7]:


# Seats won in 2010
(df[['Constituency Name', 'win_last']]
    .drop_duplicates()
    .groupby('win_last')
    .count()
    .sort_values('Constituency Name', ascending=False)
)


# In[8]:


# Seats forecast to win in 2015 using UNS model
(df[['Constituency Name', 'swing_forecast_win']]
    .drop_duplicates()
    .groupby('swing_forecast_win')
    .count()
    .sort_values('Constituency Name', ascending=False)
)


# In[9]:


# Seats actually won in 2015 (from election results)
(df[['Constituency Name', 'actual_win_now']]
    .drop_duplicates()
    .groupby('actual_win_now')
    .count()
    .sort_values('Constituency Name', ascending=False)
)


# In[10]:


# Total seat forecast accuracy = 79% of constituencies correctly predicted
(df[['Constituency Name', 'swing_forecast_win', 'actual_win_now']]
     .drop_duplicates()
     .apply(lambda row: row['swing_forecast_win'] == row['actual_win_now'], axis=1)
     .mean()
)


# In[11]:


# Total average error per party per seat = 4.45%
(df[['Constituency Name', 'actual_pc_now', 'swing_forecast_pc']]
    .apply(lambda row: abs(row['actual_pc_now'] - row['swing_forecast_pc']), axis=1)
    .mean()
)


# ### Simple ML model

# In[12]:


model = linear_model.LinearRegression()


# In[13]:


# Available features
df.columns


# In[14]:


# All features = 3.2% average error per party per seat
score_features(
    model=model,
    features=parties[1:] + \
        ['Region_' + x for x in df.Region.unique()[1:]] + \
        ['Electorate', 'Votes', 'votes_last', 'pc_last', 'won_here_last',
         'polls_now', 'swing_now', 'swing_forecast_pc']
)


# # Over to you...!

# In[15]:


# Model away...


# In[ ]:





# In[ ]:




