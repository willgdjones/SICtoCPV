
# coding: utf-8

# # SIC to CPV converter

# This is best explained I think with an example.
# 
# The Complete Leather Restoration Co. Limited is a company located in Leeds, and has a SIC code registered at companies house of http://companydb.uk/08724430-the-complete-leather-restoration-co-limited.
# 
# Given a company with SIC codes 95240, which CPV codes best represent what this company does? And using this we can query contracts finder API to find relevant contracts in their local area.
# 
# Here are the steps and results in turn which the function SICtoCPV executes:
# 
# <b>Get the descriptive title of the SIC codes</b>
# 
# `company_sic_title = lookup_sic_title(95240)`
# 
# >"Repair of furniture and home furnishings"
# 
# <b>Pull out the keywords from this title</b>
# 
# `keywords = get_keywords(company_sic_title)`
# 
# >`['repair', 'furniture', 'home', 'furnishings']`
# 
# <b>Get all CPV codes whose title mention these keywords in the description. A set of CPVs is returned for each keyword, in this case 4.</b>
# 
# `cpv_list = get_cpv_list(keywords)`
# 
# >`[[u'37414300',  u'44113700',  u'50860000',  ...  u'50884000',  u'72267000',  u'72267200'], [u'30000000', u'30100000',  u'39161000',  ...  u'45421153',  u'50850000',  u'79934000'], [u'34144800',  u'38561110',  u'45215212',  ...  u'85312200',  u'98513310'], [u'39000000', u'39143110', u'39143113', u'39516100']]`
# 
# <b>Get all the possible CPV prefixes from these sets, again return one prefix set per keyword, in this case 4.</b>
# 
# `cpv_prefixes = get_cpv_prefixes(cpv_list)`
# 
# >`[{u'37', ... u'722672'}, {u'3', ..., u'79934'}, {u'34', ..., u'9851331'}, {u'39', ..., u'395161'}]`
# 
# <b>Find out how often these CPV frequencies appear in independent keyword sets. We can see some appear once, twice and even three times.</b>
# 
# `intersection_frequencies = get_intersection_frequencies(cpv_prefixes)`
# 
# >`{u'3': 1, u'301': 1, ... , u'391431': 2, ..., u'452': 3, ... , u'5085': 2, u'9851331': 1}`
# 
# <b>Weight the importance of these CPV prefixes, multiply the frequency by how specific the CPV is.</b>
# 
# `cpv_scores = get_cpv_scores(intersection_frequencies)`
# 
# >`{u'3': 1, u'301': 3, ... , u'391431': 12, ..., u'452': 9, ... , u'5085': 8, u'9851331': 7}`
# 
# <b>Sort these CPV is order of score</b>
# 
# `top_cpvs = get_top_CPVs(cpv_scores)`
# 
# >`[(u'504114', 12),
# (u'391431', 12),
# (u'45233', 10),
# (u'39143', 10),
# ... ]`
# 
# <b>Add the titles to these return SIC codes - they mostly seem pretty relevant.</b>
# 
# `top_cpvs = add_cpv_titles(top_cpvs)`
# 
# >`[(u'Repair and maintenance services of tachometers', u'504114', 12),
# (u'Bedroom furniture', u'391431', 12),
# (u'Construction, foundation and surface works for highways, roads', u'45233', 10),
# (u'Bedroom, dining room and living-room furniture', u'39143', 10),
# ... ]`
# 
# 
# Thanks to https://github.com/nathanpitman/sic-codes for suppling the database of SIC code descriptions so I didn't have to crawl them from the raw HTML :). 

# ## Import modules

# In[2]:

import json
from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth
import requests
import pandas as pd
import nltk
nltk.data.path = ['./nltk_data']
import re
import string
import itertools
import numpy as np


# ## Helper functions

# ### Pad out SIC codes

# Pad CPV prefixes out with zeros to be length 8.

# In[3]:

def pad(cpv):
    if len(cpv) > 8:
        raise Exception('CPV too long')
    cpv += '0'* (8 - len(cpv))
    return cpv
    


# ### Tokenize CPV descriptions

# Remove punctuation marks from CPV descriptions, and set to lowercase.
# 
# `tokenize('Agricultural, farming, fishing, forestry and related products')`
# 
# `-> agricultural farming fishing forestry and related products`
# 
# 

# In[4]:

def tokenise(s):
    exclude = set(string.punctuation)
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return re.sub(regex,"",s).lower()


# ## Load CPV and SIC codes databases

# In[5]:

cabinet_api_key='o0NNKWxzI6D_JvHlDYo9Pa1l05eqnXvPwaxtk8Nx'

sic_codes = pd.read_csv('sic_codes.csv')
cpv_codes = pd.read_excel('cpv_2008_ver_2013.xlsx')
cpv_codes = cpv_codes[['CODE','EN']]
cpv_codes['CODE'] = cpv_codes['CODE'].apply(lambda x: x.split('-')[0])
sic_codes['tokenised_description'] = sic_codes['description'].apply(tokenise)



# ## Core functions

# ### Get keywords 

# Get the keywords from a given SIC code title, pulling out all the different types of verbs, nouns and adjectives.
# 
# `get_keywords('Repair of furniture and home furnishings') -> ['Repair', 'furniture', 'home', 'furnishings']`

# In[6]:

def get_keywords(company_sic_title):
    """
    Get the keywords from a SIC-title. Pulls out all types of nouns, verbs and adjectives, according to
    the nltk UPENN tagset. [https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html]
    
    No stemming (e.g. repairing -> repair) as of yet, neither does it handle exclusions. e.g. (Growing of
    vegtables EXCEPT rice).
    """
    tagged = nltk.pos_tag(company_sic_title.lower().split(' '))
    
    return [x[0] for x in tagged if x[1] in ['VBG','VB','VBD','VBN','VBP','VBZ','NNS', 'NN','NNP','NNPS','NNS','JJ','JJR','JJS'] ]


# ### Lookup SIC title

# Return the title description of a SIC codes, from the SIC code database
# 
# `lookup_sic_title('95240') -> 'Repair of furniture and home furnishings'`

# In[7]:

def lookup_sic_title(sic_code):
    """Lookup the title of a given SIC code from the SIC codes database"""
    assert isinstance(sic_codes,pd.DataFrame)
    if sic_code == '':
        return None 
    int_sic_code = int(sic_code)
    return list(sic_codes[sic_codes['code'] == int_sic_code]['description'])[0]


# ### Get CPV list

# Get the CPVs hightlighted from a list of keywords. Returns a list of lists of CPV codes equal to the number of keywords.
# 
# `get_cpvs(['repair', 'furniture', 'home', 'furnishings']) ->`
# 
# `[[u'37414300',  u'44113700',  u'50860000',  ...  u'50884000',  u'72267000',  u'72267200'], [u'30000000', u'30100000',  u'39161000',  ...  u'45421153',  u'50850000',  u'79934000'], [u'34144800',  u'38561110',  u'45215212',  ...  u'85312200',  u'98513310'], [u'39000000', u'39143110', u'39143113', u'39516100']]`

# In[8]:

### Get CPV codes
def get_cpv_list(keywords):
    """
    For a given set of keywords, return the sets of CPVs that are highlighted.
    """
    tot = []
    for kw in keywords:
        #Get the CPVs highlighted from a given keyword.
        kw_cpvs = list(cpv_codes[cpv_codes['EN'].apply(tokenise).apply(lambda x: kw in x)]['CODE'])        
        tot.append(kw_cpvs)
    return tot



# ### Get CPV prefix sets

# For list of CPVs, return all the unique prefixes of the CPVs.
# 
# `create_cpv_sets([[u'44113700', u'44167200']]) -> [{u'44', u'441', u'4411', u'44113', u'441137', u'4416', u'44167', u'441672'}]`

# In[9]:

def get_cpv_prefixes(cpv_list):
    cpv_sets = []

    for l in cpv_list:
        cpvs = []
        for c in l:
            for i in range(2,9):
                cpvs.append(c[0:i])
        cpvs = [x.rstrip('0') for x in cpvs]
        cpv_sets.append(set(cpvs))
    return cpv_sets


# ### Create intersection lists

# For a list of CPV prefix sets, get the frequencies of how often they appear in different cpv_sets.
# 
# `get_intersection_frequencies([set(['37','375']), set(['37', '42'])]) -> {'37': 2, '375': 1, '42': 1}`

# In[10]:

def get_intersection_frequencies(cpv_prefixes):
    
    intersection_lists = []
    for s in range(1,len(cpv_prefixes)+1):
        for i in itertools.combinations(range(0,len(cpv_prefixes)),s):
            intersection_lists.append([i, list(set.intersection(*[cpv_prefixes[j] for j in i]))])
    intersection_frequencies = {}

    for l, interscts in intersection_lists:
        for i in interscts:
            if i in intersection_frequencies:
                if len(l) > intersection_frequencies[i]:
                    intersection_frequencies[i] = len(l)
            else:
                intersection_frequencies[i] = len(l)
                
    return intersection_frequencies


# ### Score CPV codes

# Given the frequency of a CPV codes, combine this with it' specificity to give it a score.

# In[11]:

def get_cpv_scores(intersection_frequencies):
    
    cpv_scores = {}
    for k,v in intersection_frequencies.items():
        intersection_frequency = v
        specificity = len(k)
        cpv_scores[k] = scoring_heuristic(intersection_frequency, specificity)

    return cpv_scores


# ### Get top CPVs 

# Return the list of CPVs ordered by their scores

# In[12]:

def get_top_CPVs(cpv_scores):
    top_cpvs = [ (k,v) for (k,v) in cpv_scores.items() ]
    top_cpvs.sort(key=lambda x: x[1], reverse=True)
    return top_cpvs


# ### Scoring heuristic 

# The scoring heurstic is the key component of this process. Given a cpv codes with a known intersection_frequency and specificity, what score should we give it? How important is it that it is specific, and how important is it that it is highlighted by multiple keywords? Right now I just multiply them together :o).

# In[13]:

def scoring_heuristic(intersection_frequency, specificity):
    return intersection_frequency * specificity
    


# ### Add CPV titles

# Add the titles of a list of CPV codes

# In[14]:

def add_cpv_titles(top_cpvs):
    
    cpv_titles = [cpv_codes[cpv_codes['CODE'] == pad(cpv[0])]['EN'].values[0] for cpv in top_cpvs]
    return [(cpv_titles[i],) + top_cpvs[i] for i in range(len(top_cpvs))]


# ### SICtoCPV

# The main function to export. Takes in a SIC code, and outputs relevant CPV codes plus the titles.

# In[15]:

def SICtoCPV(sic_code):
    
    company_sic_title = lookup_sic_title(sic_code)

    keywords = get_keywords(company_sic_title)

    cpv_list = get_cpv_list(keywords)

    cpv_prefixes = get_cpv_prefixes(cpv_list)

    intersection_frequencies = get_intersection_frequencies(cpv_prefixes)

    cpv_scores = get_cpv_scores(intersection_frequencies)

    top_cpvs = get_top_CPVs(cpv_scores)

    top_cpvs = add_cpv_titles(top_cpvs)
    
    return [(sic_code, company_sic_title), top_cpvs]


# https://github.com/nathanpitman/sic-codes
