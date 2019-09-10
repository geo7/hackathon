#!/usr/bin/env python
# coding: utf-8

# # SixFifty GE2017 Model
# ## Create modelling datasets
# For more information please see [SixFifty.org.uk](https://sixfifty.org.uk) or the [SixFifty Hackathon repo](https://github.com/six50/hackathon).

# ## Import datasets and pre-flight checks

# In[1]:


# Libaries
import feather
import matplotlib
import numpy as np
from pathlib import Path
import pandas as pd

# Config
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
DATA_DIR = Path('../data/')
get_ipython().run_line_magic('matplotlib', 'inline')


# Running `data/retrieve_data.py` from the `hackathon` repo root will download the following datasets into the required location.

# In[2]:


ge_data_dir = DATA_DIR / 'general_election' / 'electoral_commission' / 'results'
ge_2010 = pd.read_feather(ge_data_dir / 'ge_2010_results.feather')
ge_2015 = pd.read_feather(ge_data_dir / 'ge_2015_results.feather')


# In[3]:


ge_2010.head(3)


# In[4]:


ge_2015.head(3)


# Before we get into the modelling, there's a bit of cleanup to do and a few checks to run:
# - [Some MPs](https://en.wikipedia.org/wiki/Labour_Co-operative) are members of _both_ the [Labour Party](www.labour.org.uk) and the [Co-operative Party](http://www.party.coop/), which plays havoc with modelling. We will therefore consider them all members of the Labour party.
# - Check that there are all constituencies in the 2010 data are also in 2015 data, and vice versa.
# - Create `country_lookup`, a dictionary that returns the country of any constituency given its PANO (Press Association ID Number).

# In[5]:


parties_lookup_2010 = {
    'Con': 'con',
    'Lab': 'lab',
    'LD': 'ld',
    'UKIP': 'ukip',
    'Grn': 'grn',
    'Other': 'other'
}
parties_15 = list(parties_lookup_2010.values())


# In[6]:


parties_lookup_2015 = {
    'C': 'con',
    'Lab': 'lab',
    'LD': 'ld',
    'UKIP': 'ukip',
    'Green': 'grn',
    'SNP': 'snp',
    'PC': 'pc',
    'Other': 'other'
}
parties_17 = list(parties_lookup_2015.values())


# In[7]:


# Merge Labour and Coop
ge_2015['Lab'] = ge_2015['Lab'] + ge_2015['Lab Co-op']
del ge_2015['Lab Co-op']


# In[8]:


# Check constituencies are mergeable
print(set(ge_2010['Press Association Reference']).difference(set(ge_2015['Press Association ID Number'])))  # should be empty set
print(set(ge_2015['Press Association ID Number']).difference(set(ge_2010['Press Association Reference'])))  # should be empty set
print(len(ge_2010), len(ge_2010['Press Association Reference']))  # should both be 650
print(len(ge_2015), len(ge_2015['Press Association ID Number']))  # should both be 650


# In[9]:


# Make PANO -> geo lookup
geo_lookup = [(x[1][0], x[1][1]) for x in ge_2015[['Press Association ID Number', 'Country']].iterrows()]
geo_lookup = dict(geo_lookup)
print(geo_lookup[14.0])  # should be "Northern Ireland"
# Add London boroughs
london_panos = ge_2015[ge_2015['County'] == 'London']['Press Association ID Number'].values
for pano in london_panos:
    geo_lookup[pano] = 'London'
print(geo_lookup[237.0])  # should be "London"
# Rename other England
for k in geo_lookup:
    if geo_lookup[k] == 'England':
        geo_lookup[k] = 'England_not_london'
    elif geo_lookup[k] == 'Northern Ireland':
        geo_lookup[k] = 'NI'


# ## 2015 polling

# In[10]:


polls_data_dir = DATA_DIR / 'polls'


# In[11]:


# Latest polling data (3 days before election, i.e. if election on 7th May 2015, polls as of 4th May)
polls = pd.read_feather(polls_data_dir / 'polls.feather')
polls.head()


# In[12]:


pollsters = polls[(polls.to >= '2015-04-04') & (polls.to <= '2015-05-04')].company.unique()
pollsters


# In[13]:


# Use single last poll from each pollster in final week of polling then average out
polls = polls[(polls.to >= '2015-04-01') & (polls.to <= '2015-05-07')]
pop = polls.loc[:0]
for p in pollsters:
    pop = pop.append(polls[polls.company == p].tail(1))
pop


# In[14]:


# Create new polls dictionary by geo containing simple average across all pollsters
polls = {'UK': {}}
for p in ['con', 'lab', 'ld', 'ukip', 'grn']:
    polls['UK'][p] = pop[p].mean()
polls['UK'] = pd.Series(polls['UK'])
polls['UK']


# In[15]:


# Scotland, Wales, NI, London not available in 2015 data (we haven't extracted them yet!)
# Add Other
for geo in ['UK']:
    if 'other' not in polls[geo]:
        polls[geo]['other'] = 1 - sum(polls[geo])


# In[16]:


# Reweight to 100%
for geo in ['UK']:
    polls[geo] = polls[geo] / polls[geo].sum()
polls


# ## 2017 polling

# In[17]:


# Latest polling data
polls_17 = {'UK': {}}
polls_17_uk = pd.read_feather(polls_data_dir / 'polls.feather')
# Filter to recent data
polls_17_uk = polls_17_uk[polls_17_uk.to >= '2017-06-06']
# Add parties
for p in ['con', 'lab', 'ld', 'ukip', 'grn', 'snp']:
    polls_17['UK'][p] = (polls_17_uk.sample_size * polls_17_uk[p]).sum() / polls_17_uk.sample_size.sum()
polls_17['UK'] = pd.Series(polls_17['UK'], index=['con', 'lab', 'ld', 'ukip', 'snp', 'grn'])
polls_17


# In[18]:


# Repeat for Scotland polling...
polls_17['Scotland'] = {}
polls_17_tmp = pd.read_feather(polls_data_dir / 'polls_scotland.feather')
polls_17_tmp = polls_17_tmp[polls_17_tmp.to >= '2017-06-05']
for p in ['con', 'lab', 'ld', 'ukip', 'snp', 'grn']:
    polls_17['Scotland'][p] = (polls_17_tmp.sample_size * polls_17_tmp[p]).sum() / polls_17_tmp.sample_size.sum()
polls_17['Scotland'] = pd.Series(polls_17['Scotland'], index=['con', 'lab', 'ld', 'ukip', 'snp', 'grn'])
polls_17['Scotland']


# In[19]:


# ...and Wales
polls_17['Wales'] = {}
polls_17_tmp = pd.read_feather(polls_data_dir / 'polls_wales.feather')
polls_17_tmp = polls_17_tmp[polls_17_tmp.to >= '2017-06-07']
for p in ['con', 'lab', 'ld', 'ukip', 'pc', 'grn']:
    polls_17['Wales'][p] = (polls_17_tmp.sample_size * polls_17_tmp[p]).sum() / polls_17_tmp.sample_size.sum()
polls_17['Wales'] = pd.Series(polls_17['Wales'], index=['con', 'lab', 'ld', 'ukip', 'pc', 'grn'])
polls_17['Wales']


# In[20]:


# NI
polls_17['NI'] = (pd.read_feather(polls_data_dir / 'polls_ni_smoothed.feather')
                    .sort_values(by='date', ascending=False).iloc[0])
del polls_17['NI']['date']

# Collate all NI parties under other
for k in polls_17['NI'].index:
    if k not in parties_17:
        del polls_17['NI'][k]

del polls_17['NI']['other']
polls_17['NI']


# In[21]:


# London
polls_17['London'] = {}
polls_17_tmp = pd.read_feather(polls_data_dir / 'polls_london.feather')
polls_17_tmp = polls_17_tmp[polls_17_tmp.to >= '2017-05-31']
for p in ['con', 'lab', 'ld', 'ukip', 'grn']:
    polls_17['London'][p] = (polls_17_tmp.sample_size * polls_17_tmp[p]).sum() / polls_17_tmp.sample_size.sum()
polls_17['London'] = pd.Series(polls_17['London'], index=['con', 'lab', 'ld', 'ukip', 'grn'])
polls_17['London']


# In[22]:


# Estimate polling for England excluding London
survation_wts = {
    # from http://survation.com/wp-content/uploads/2017/06/Final-MoS-Post-BBC-Event-Poll-020617SWCH-1c0d4h9.pdf
    'Scotland': 85,
    'England': 881,
    'Wales': 67,
    'London': 137,
    'NI': 16
}
survation_wts['England_not_london'] = survation_wts['England'] - survation_wts['London']
survation_wts['UK'] = survation_wts['Scotland'] + survation_wts['England'] + survation_wts['Wales'] + survation_wts['NI']

def calculate_england_not_london(party):
    out = polls_17['UK'][party] * survation_wts['UK']
    for geo in ['Scotland', 'Wales', 'NI', 'London']:
        if party in polls_17[geo]:
            out = out - polls_17[geo][party] * survation_wts[geo]
    out = out / survation_wts['England_not_london']
    return out

polls_17['England_not_london'] = {'pc': 0, 'snp': 0}
for party in ['con', 'lab', 'ld', 'ukip', 'grn']:
    polls_17['England_not_london'][party] = calculate_england_not_london(party)

polls_17['England_not_london'] = pd.Series(polls_17['England_not_london'])
polls_17['England_not_london']


# In[23]:


# Fill in the gaps
for geo in ['UK', 'Scotland', 'Wales', 'NI', 'London', 'England_not_london']:
    for party in ['con', 'lab', 'ld', 'ukip', 'grn', 'snp', 'pc']:
        if party not in polls_17[geo]:
            print("Adding {} to {}".format(party, geo))
            polls_17[geo][party] = 0


# In[24]:


# Fix PC (Plaid Cymru) for UK
polls_17['UK']['pc'] = polls_17['Wales']['pc'] * survation_wts['Wales'] / survation_wts['UK']


# In[25]:


# Add Other
for geo in ['UK', 'Scotland', 'Wales', 'NI', 'London', 'England_not_london']:
    if 'other' not in polls_17[geo]:
        polls_17[geo]['other'] = 1 - sum(polls_17[geo])

# This doesn't work for UK or England_not_london; set current other polling to match 2015 result
polls_17['UK']['other'] = 0.03 # ge.other.sum() / ge['Valid Votes'].sum()
polls_17['England_not_london']['other'] = 0.01 # ge[ge.geo == 'England_not_london'].other.sum() / ge[ge.geo == 'England_not_london']['Valid Votes'].sum()


# In[26]:


# Reweight to 100%
for geo in ['UK', 'Scotland', 'Wales', 'NI', 'London', 'England_not_london']:
    polls_17[geo] = polls_17[geo] / polls_17[geo].sum()


# In[27]:


# Let's take a look!
polls_17


# ## Export polling data

# In[28]:


polls_15_csv = pd.DataFrame(columns=['con', 'lab', 'ld', 'ukip', 'grn', 'snp', 'pc', 'other'])
for geo in polls:
    for party in polls[geo].index:
        polls_15_csv.loc[geo, party] = polls[geo].loc[party]
polls_15_csv.to_csv(polls_data_dir / 'final_polls_2015.csv', index=True)
polls_15_csv


# In[29]:


polls_17_csv = pd.DataFrame(columns=['con', 'lab', 'ld', 'ukip', 'grn', 'snp', 'pc', 'other'])
for geo in polls_17:
    for party in polls_17[geo].index:
        polls_17_csv.loc[geo, party] = polls_17[geo].loc[party]
polls_17_csv.to_csv(polls_data_dir / 'final_polls_2017.csv', index=True)
polls_17_csv


# ## Reduce ge_2010 dataframe to above results only

# In[30]:


# GE 2010 dataset has a lot of parties...
ge_2010.head()


# In[31]:


# Top 15 parties
ge_2010.iloc[:, 11:].sum().sort_values(ascending=False).head(15)


# In[32]:


# Define other parties
other_parties = list(set(ge_2010.columns) - set(ge_2010.columns[:6]) - set(parties_lookup_2010.keys()))
other_parties_2015 = list(set(ge_2015.columns) - set(ge_2015.columns[:11]) - set(parties_lookup_2015.keys()))

ge_2010['Other'] = ge_2010.loc[:, other_parties].sum(axis=1)
ge_2015['Other'] = ge_2015.loc[:, other_parties_2015].sum(axis=1)


# ### # Export somewhat cleaned up 2010/2015 results data

# In[33]:


# It looks like this
ge_2010.head(3)


# In[34]:


# Export to disk
ge_2010.to_csv(DATA_DIR / 'model' / 'ge10_all_parties.csv', index=False)
ge_2010.to_feather(DATA_DIR / 'model' / 'ge10_all_parties.feather')

ge_2015.to_csv(DATA_DIR / 'model' / 'ge15_all_parties.csv', index=False)
ge_2015.to_feather(DATA_DIR / 'model' / 'ge15_all_parties.feather')


# ### Rename for convenience

# In[35]:


ge_2010_full = ge_2010.copy()
ge_2015_full = ge_2015.copy()


# ### Filter to metadata cols + parties of interest

# In[36]:


parties_15 = ['con', 'lab', 'ld', 'ukip', 'grn', 'other']
parties_17 = ['con', 'lab', 'ld', 'ukip', 'grn', 'snp', 'pc', 'other']

parties_lookup_2010 = {
    'Con': 'con',
    'Lab': 'lab',
    'LD': 'ld',
    'UKIP': 'ukip',
    'Grn': 'grn',
    'Other': 'other'
}

parties_lookup_2015 = {
    'C': 'con',
    'Lab': 'lab',
    'LD': 'ld',
    'UKIP': 'ukip',
    'Green': 'grn',
    'SNP': 'snp',
    'PC': 'pc',
    'Other': 'other'
}


# In[37]:


# Filter ge to metadata cols + parties of interest
ge_2010 = ge_2010.loc[:, list(ge_2010.columns[:6]) + list(parties_lookup_2010.keys())]
ge_2015 = ge_2015.loc[:, list(ge_2015.columns[:11]) + list(parties_lookup_2015.keys())]

# Rename parties
ge_2010.columns = [parties_lookup_2010[x] if x in parties_lookup_2010 else x for x in ge_2010.columns]
ge_2015.columns = [parties_lookup_2015[x] if x in parties_lookup_2015 else x for x in ge_2015.columns]

# Calculate vote share
for party in parties_15:
    ge_2010[party + '_pc'] = ge_2010[party] / ge_2010['Votes']

for party in parties_17:
    ge_2015[party + '_pc'] = ge_2015[party] / ge_2015['Valid Votes']

ge_2010.head(3)


# In[38]:


# Export to disk
ge_2010.to_csv(DATA_DIR / 'model' / 'ge10.csv', index=False)
ge_2010.to_feather(DATA_DIR / 'model' / 'ge10.feather')

ge_2015.to_csv(DATA_DIR / 'model' / 'ge15.csv', index=False)
ge_2015.to_feather(DATA_DIR / 'model' / 'ge15.feather')


# ### Calculate uplifts ("swing")

# In[39]:


# Calculate national voteshare in 2010
ge_2010_totals = ge_2010.loc[:, ['Votes'] + parties_15].sum()
ge_2010_voteshare = ge_2010_totals / ge_2010_totals['Votes']
del ge_2010_voteshare['Votes']
ge_2010_voteshare


# In[40]:


# Calculate swing between 2015 and latest smoothed polling
swing = ge_2010_voteshare.copy()
for party in parties_15:
    swing[party] = polls_15_csv.loc['UK', party] / ge_2010_voteshare[party] - 1
    ge_2010[party + '_swing'] = polls_15_csv.loc['UK', party] / ge_2010_voteshare[party] - 1
swing


# In[41]:


# Forecast is previous result multiplied by swing uplift
for party in parties_15:
    ge_2010[party + '_forecast'] = ge_2010[party + '_pc'] * (1 + swing[party])


# In[42]:


def win_10(row):
    all_parties = set(ge_2010_full.columns[6:]) - set(['Other'])
    out = row[all_parties].sort_values(ascending=False).index[0]
    if out in parties_lookup_2010.keys():
        out = parties_lookup_2010[out]
    elif out == 'Speaker':
        out = 'other'
    return out

def win_15(row):
    all_parties = set(ge_2015_full.columns[11:]) - set(['Other'])
    out = row[all_parties].sort_values(ascending=False).index[0]
    if out in parties_lookup_2015.keys():
        out = parties_lookup_2015[out]
    elif out == 'Speaker':
        out = 'other'
    return out


# In[43]:


def pred_15(row):
    return row[[p + '_forecast' for p in parties_15]].sort_values(ascending=False).index[0].replace('_forecast', '')


# In[44]:


ge_2010['win_10'] = ge_2010_full.apply(win_10, axis=1)
ge_2015['win_15'] = ge_2015_full.apply(win_15, axis=1)


# In[45]:


ge_2010['win_15'] = ge_2010.apply(pred_15, axis=1)


# In[46]:


ge_2010.groupby('win_10').count()['Constituency Name'].sort_values(ascending=False)


# In[47]:


ge_2010.groupby('win_15').count()['Constituency Name'].sort_values(ascending=False)


# In[48]:


ge_2015.groupby('win_15').count()['Constituency Name'].sort_values(ascending=False)


# ### Calculate Geo-Level Voteshare + Swing inc. all parties

# In[49]:


# Add geos
ge_2015['geo'] = ge_2015['Press Association ID Number'].map(geo_lookup)
geos = list(ge_2015.geo.unique())


# In[50]:


# Calculate geo-level voteshare in 2015
ge_2015_totals = ge_2015.loc[:, ['Valid Votes', 'geo'] + parties_17].groupby('geo').sum()
ge_2015_totals


# In[51]:


# Convert into vote share
ge_2015_voteshare = ge_2015_totals.div(ge_2015_totals['Valid Votes'], axis=0)
del ge_2015_voteshare['Valid Votes']
ge_2015_voteshare


# In[52]:


# Calculate geo-swing
swing_17 = ge_2015_voteshare.copy()
for party in parties_17:
    for geo in geos:
        if ge_2015_voteshare.loc[geo][party] > 0:
            out = polls_17[geo][party] / ge_2015_voteshare.loc[geo][party] - 1
        else:
            out = 0.0
        swing_17.loc[geo, party] = out

swing_17


# In[53]:


# Apply swing
for party in parties_17:
    ge_2015[party + '_swing'] = ge_2015.apply(lambda row: swing_17.loc[row['geo']][party], axis=1)
    ge_2015[party + '_2017_forecast'] = ge_2015.apply(lambda x: x[party + '_pc'] * (1 + swing_17.loc[x['geo']][party]), axis=1)


# In[54]:


ge_2015.groupby('win_15').count()['Constituency Name'].sort_values(ascending=False)


# In[55]:


def win_17(row):
    return row[[p + '_2017_forecast' for p in parties_17]].sort_values(ascending=False).index[0].replace('_2017_forecast', '')


# In[56]:


ge_2015['win_17'] = ge_2015.apply(win_17, axis=1)


# In[57]:


ge_2015.groupby('win_17').count()['Constituency Name'].sort_values(ascending=False)


# ### Turn into ML-ready dataset

# In[58]:


parties = ['con', 'lab', 'ld', 'ukip', 'grn']


# In[59]:


act_15_lookup = {k: v for i, (k, v) in ge_2015[['Press Association ID Number', 'win_15']].iterrows()}
ge_2010['act_15'] = ge_2010['Press Association Reference'].map(act_15_lookup)


# In[60]:


pc_15_lookup = {
    p: {k: v for i, (k, v) in ge_2015[['Press Association ID Number', p + '_pc']].iterrows()} for p in parties
}


# In[61]:


for p in parties:
    ge_2010[p + '_actual'] = ge_2010['Press Association Reference'].map(pc_15_lookup[p])


# ### Melt into following cols:
# - 'Press Association Reference'
# - 'Constituency Name'
# - 'Region'
# - 'Electorate'
# - 'Votes'
# - 'party'
# - 'votes_last'
# - 'pc_last'
# - 'win_last'
# - 'polls_now'
# - 'swing_now'
# - 'swing_forecast_pc'
# - 'swing_forecast_win'
# - 'actual_win_now'
# - 'actual_pc_now'

# In[62]:


df = ge_2010[['Press Association Reference', 'Constituency Name', 'Region', 'Electorate', 'Votes'] + parties]
df.head()


# In[63]:


df.shape


# In[64]:


df = pd.melt(
    df,
    id_vars=['Press Association Reference', 'Constituency Name', 'Region', 'Electorate', 'Votes'],
    value_vars=parties,
    var_name='party',
    value_name='votes_last'
)


# In[65]:


df.shape


# In[66]:


# pc_last


# In[67]:


pc_last = pd.melt(
    ge_2010[['Press Association Reference'] + [p + '_pc' for p in parties]],
    id_vars=['Press Association Reference'],
    value_vars=[p + '_pc' for p in parties],
    var_name='party',
    value_name='pc_last'
)
pc_last['party'] = pc_last.party.apply(lambda x: x.replace('_pc', ''))


# In[68]:


df = pd.merge(
    left=df,
    right=pc_last,
    how='left',
    on=['Press Association Reference', 'party']
)


# In[69]:


df.head(3)


# In[70]:


# win_last


# In[71]:


win_last = ge_2010[['Press Association Reference', 'win_10']]
win_last.columns = ['Press Association Reference', 'win_last']
df = pd.merge(
    left=df,
    right=win_last,
    on=['Press Association Reference']
)


# In[72]:


df.head(3)


# In[73]:


# polls_now
df['polls_now'] = df.party.map(polls['UK'])
df.head(3)


# In[74]:


# swing_now
swing_now = pd.melt(
    ge_2010[['Press Association Reference'] + [p + '_swing' for p in parties]],
    id_vars=['Press Association Reference'],
    value_vars=[p + '_swing' for p in parties],
    var_name='party',
    value_name='swing_now'
)
swing_now['party'] = swing_now.party.apply(lambda x: x.replace('_swing', ''))

df = pd.merge(
    left=df,
    right=swing_now,
    how='left',
    on=['Press Association Reference', 'party']
)
df.head(10)


# In[75]:


# swing_forecast_pc
swing_forecast_pc = pd.melt(
    ge_2010[['Press Association Reference'] + [p + '_forecast' for p in parties]],
    id_vars=['Press Association Reference'],
    value_vars=[p + '_forecast' for p in parties],
    var_name='party',
    value_name='swing_forecast_pc'
)
swing_forecast_pc['party'] = swing_forecast_pc.party.apply(lambda x: x.replace('_forecast', ''))

df = pd.merge(
    left=df,
    right=swing_forecast_pc,
    how='left',
    on=['Press Association Reference', 'party']
)


# In[76]:


df.head(3)


# In[77]:


# swing_forecast_win
swing_forecast_win = ge_2010[['Press Association Reference', 'win_15']]
swing_forecast_win.columns = ['Press Association Reference', 'swing_forecast_win']
df = pd.merge(
    left=df,
    right=swing_forecast_win,
    on=['Press Association Reference']
)


# In[78]:


df.head(3)


# In[79]:


# actual_win_now
actual_win_now = ge_2010[['Press Association Reference', 'act_15']]
actual_win_now.columns = ['Press Association Reference', 'actual_win_now']
df = pd.merge(
    left=df,
    right=actual_win_now,
    on=['Press Association Reference']
)
df.head(3)


# In[80]:


# actual_pc_now
actual_pc_now = pd.melt(
    ge_2010[['Press Association Reference'] + [p + '_actual' for p in parties]],
    id_vars=['Press Association Reference'],
    value_vars=[p + '_actual' for p in parties],
    var_name='party',
    value_name='actual_pc_now'
)
actual_pc_now['party'] = actual_pc_now.party.apply(lambda x: x.replace('_actual', ''))

df = pd.merge(
    left=df,
    right=actual_pc_now,
    how='left',
    on=['Press Association Reference', 'party']
)


# In[81]:


df.head(3)


# In[82]:


# dummy party
df = pd.concat([df, pd.get_dummies(df.party)], axis=1)

# dummy region
df = pd.concat([df, pd.get_dummies(df.Region, prefix='Region')], axis=1)

df.head(3)


# In[83]:


# won_here_last
df['won_here_last'] = (df['party'] == df['win_last']).astype('int')


# In[84]:


df.head(20)


# ### Export final 2010 -> 2015 training set

# In[85]:


df.to_csv(DATA_DIR / 'model' / 'ge_2010_2015_training_data.csv', index=False)
df.to_feather(DATA_DIR / 'model' / 'ge_2010_2015_training_data.feather')


# ### Recreate this training dataset using same column names for 2015 -> 2017 for a GE2017 forecast
# This would be much simpler with a bit of refactoring! Anyone interested in helping out please get in touch with @john_sandall, happy to give free mentoring in return for collaborating on this work!

# ### Melt into following cols:
# - 'Press Association Reference'
# - 'Constituency Name'
# - 'Region'
# - 'Electorate'
# - 'Votes'
# - 'party'
# - 'votes_last'
# - 'pc_last'
# - 'win_last'
# - 'polls_now'
# - 'swing_now'
# - 'swing_forecast_pc'
# - 'swing_forecast_win'

# In[87]:


# Add SNP and Plaid Cymru
parties += ['snp', 'pc']
parties


# In[88]:


df15 = ge_2015[['Press Association ID Number', 'Constituency Name', 'Region', 'geo', 'Electorate', 'Valid Votes'] + parties]
df15.columns = ['Press Association ID Number', 'Constituency Name', 'Region', 'geo', 'Electorate', 'Votes'] + parties
df15.head()


# In[89]:


df15.shape


# In[90]:


df15 = pd.melt(
    df15,
    id_vars=['Press Association ID Number', 'Constituency Name', 'Region', 'geo', 'Electorate', 'Votes'],
    value_vars=parties,
    var_name='party',
    value_name='votes_last'
)


# In[91]:


df15.shape


# In[93]:


# pc_last
pc_last = pd.melt(
    ge_2015[['Press Association ID Number'] + [p + '_pc' for p in parties]],
    id_vars=['Press Association ID Number'],
    value_vars=[p + '_pc' for p in parties],
    var_name='party',
    value_name='pc_last'
)
pc_last['party'] = pc_last.party.apply(lambda x: x.replace('_pc', ''))


# In[94]:


df15 = pd.merge(
    left=df15,
    right=pc_last,
    how='left',
    on=['Press Association ID Number', 'party']
)


# In[95]:


# win_last
win_last = ge_2015[['Press Association ID Number', 'win_15']]
win_last.columns = ['Press Association ID Number', 'win_last']
df15 = pd.merge(
    left=df15,
    right=win_last,
    on=['Press Association ID Number']
)


# In[96]:


# polls_now <- USE REGIONAL POLLING! (Possibly a very bad idea, the regional UNS performed worse than national!)
df15['polls_now'] = df15.apply(lambda row: polls_17[row.geo][row.party], axis=1)


# In[97]:


# swing_now
swing_now = pd.melt(
    ge_2015[['Press Association ID Number'] + [p + '_swing' for p in parties]],
    id_vars=['Press Association ID Number'],
    value_vars=[p + '_swing' for p in parties],
    var_name='party',
    value_name='swing_now'
)
swing_now['party'] = swing_now.party.apply(lambda x: x.replace('_swing', ''))

df15 = pd.merge(
    left=df15,
    right=swing_now,
    how='left',
    on=['Press Association ID Number', 'party']
)


# In[98]:


# swing_forecast_pc
swing_forecast_pc = pd.melt(
    ge_2015[['Press Association ID Number'] + [p + '_2017_forecast' for p in parties]],
    id_vars=['Press Association ID Number'],
    value_vars=[p + '_2017_forecast' for p in parties],
    var_name='party',
    value_name='swing_forecast_pc'
)
swing_forecast_pc['party'] = swing_forecast_pc.party.apply(lambda x: x.replace('_2017_forecast', ''))

df15 = pd.merge(
    left=df15,
    right=swing_forecast_pc,
    how='left',
    on=['Press Association ID Number', 'party']
)


# In[99]:


# swing_forecast_win
swing_forecast_win = ge_2015[['Press Association ID Number', 'win_17']]
swing_forecast_win.columns = ['Press Association ID Number', 'swing_forecast_win']
df15 = pd.merge(
    left=df15,
    right=swing_forecast_win,
    on=['Press Association ID Number']
)


# In[100]:


# dummy party
df15 = pd.concat([df15, pd.get_dummies(df15.party)], axis=1)


# In[101]:


# dummy region
df15 = pd.concat([df15, pd.get_dummies(df15.Region, prefix='Region')], axis=1)


# In[102]:


# won_here_last
df15['won_here_last'] = (df15['party'] == df15['win_last']).astype('int')


# In[103]:


df15.head(20)


# ### Export final 2015 -> 2017 prediction set

# In[104]:


df15.to_csv(DATA_DIR / 'model' / 'ge_2015_2017_prediction_data.csv', index=False)
df15.to_feather(DATA_DIR / 'model' / 'ge_2015_2017_prediction_data.feather')

