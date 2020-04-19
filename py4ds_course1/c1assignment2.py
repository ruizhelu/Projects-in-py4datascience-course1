#Assignment 2 - Pandas Introduction
#All questions are weighted the same in this assignment.

#Part 1
#The following code loads the olympics dataset (olympics.csv), which was derrived from the Wikipedia entry on All Time Olympic Games Medals, and does some basic data cleaning.

#The columns are organized as # of Summer games, Summer medals, # of Winter games, Winter medals, total # number of games, total # of medals. Use this dataset to answer the questions below.

import pandas as pd

df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index)
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
df.head()

#Question 1
#Which country has won the most gold medals in summer games?

#This function should return a single string value.

#Question1:
def answer_one():
    df['Country'] = df.index
    df1 = df.set_index(['Gold'])
    df1 = df1.sort_index()
    return df1.iloc[-1]['Country']
answer_one()

#Question 2
#Which country had the biggest difference between their summer and winter gold medal counts?

#This function should return a single string value.

#Question2:
def answer_two():
    return (df['Gold'] - df['Gold.1']).idxmax()
answer_two()

or

def answer_two():
    diff = df['Gold'] - df['Gold.1']
    c = diff[diff == max(diff)].index[0]
    return c
answer_two()

#Question 3
#Which country has the biggest difference between their summer gold medal counts and winter gold medal counts relative to their total gold medal count?

#(Summer Gold − Winter Gold)/(Total Gold)

#Only include countries that have won at least 1 gold in both summer and winter.

#This function should return a single string value.

#Question3:
def answer_three():
    onegsw = df.where((df['Gold'] > 0) & (df['Gold.1'] > 0))
    diff = onegsw['Gold'] - onegsw['Gold.1']
    summ = onegsw['Gold.2']
    rate = diff/summ
    return rate.idxmax()
answer_three()

or

def answer_three():
    f1 = df[df['Gold']>0]
    f2 = f1[f1['Gold.1']>0]
    summer = f2['Gold']
    winter = f2['Gold.1']
    total = summer + winter
    relative = (summer - winter) / total
    return relative[relative == max(relative)].index[0]
answer_three()

#Question 4
#Write a function that creates a Series called "Points" which is a weighted value where each gold medal (Gold.2) counts for 3 points, silver medals (Silver.2) for 2 points, and bronze medals (Bronze.2) for 1 point. The function should return only the column (a Series object) which you created, with the country names as indices.

#This function should return a Series named Points of length 146

#Question4:
def answer_four():
    df['Points'] = df['Gold.2']*3 + df['Silver.2']*2 + df['Bronze.2']
    return df['Points']
answer_four()

#Part 2
#For the next set of questions, we will be using census data from the United States Census Bureau. Counties are political and geographic subdivisions of states in the United States. This dataset contains population data for counties and states in the US from 2010 to 2015. See this document for a description of the variable names.

#The census dataset (census.csv) should be loaded as census_df. Answer questions using this as appropriate.

#Question 5
#Which state has the most counties in it? (hint: consider the sumlevel key carefully! You'll need this for future questions too...)

#This function should return a single string value.

#Question5:
def answer_five():
    df = census_df.where(census_df['SUMLEV'] == 50)
    df = df.groupby( [ 'SUMLEV', 'STNAME'] ).size()
    c = df.idxmax()[1]
    return c
answer_five()

#Question 6
#Only looking at the three most populous counties for each state, what are the three most populous states (in order of highest population to lowest population)? Use CENSUS2010POP.

#This function should return a list of string values.

#Question6:
def answer_six():
    df = census_df.where(census_df['SUMLEV'] == 50)
    df = df.sort(['STNAME','CENSUS2010POP'],ascending=False).groupby('STNAME').head(3)
    df = df.reset_index().groupby('STNAME').sum()
    top3 = df.sort(['CENSUS2010POP'],ascending=False).head(3)
    return list(top3.index.values)
answer_six()

#Question 7
#Which county has had the largest absolute change in population within the period 2010-2015? (Hint: population values are stored in columns POPESTIMATE2010 through POPESTIMATE2015, you need to consider all six columns.)

#e.g. If County Population in the 5 year period is 100, 120, 80, 105, 100, 130, then its largest change in the period would be |130-80| = 50.

#This function should return a single string value.

#Question7:
def answer_seven():
    df = census_df.where(census_df['SUMLEV'] == 50)
    q7 = ['STNAME','CTYNAME','POPESTIMATE2010','POPESTIMATE2011','POPESTIMATE2012','POPESTIMATE2013','POPESTIMATE2014','POPESTIMATE2015']
    df = df[q7].set_index(['STNAME','CTYNAME']).stack()
    df = df.max(level=['STNAME','CTYNAME']) - df.min(level=['STNAME','CTYNAME'])
    c = df.idxmax()[1]
    return c
answer_seven()

#Question 8
#In this datafile, the United States is broken up into four regions using the "REGION" column.

#Create a query that finds the counties that belong to regions 1 or 2, whose name starts with 'Washington', and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.

#This function should return a 5x2 DataFrame with the columns = ['STNAME', 'CTYNAME'] and the same index ID as the census_df (sorted ascending by index).

#Question8:
def answer_eight():
    df = census_df[census_df['SUMLEV'] == 50]
    df = df.where((df['REGION'] == 1) | (df['REGION'] == 2))
    df = df.where((df['POPESTIMATE2015']) > (df['POPESTIMATE2014'])).dropna()
    df = df[df['CTYNAME'].str.startswith("Washington")]
    df = df[['STNAME','CTYNAME']]
    return df
answer_eight()
