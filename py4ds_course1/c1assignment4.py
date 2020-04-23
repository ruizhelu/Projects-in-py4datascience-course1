#Assignment 4 - Hypothesis Testing
#This assignment requires more individual learning than previous assignments - you are encouraged to check out the pandas documentation to find functions or methods you might not have used yet, or ask questions on Stack Overflow and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.

#Definitions:

#A quarter is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
#A recession is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
#A recession bottom is the quarter within a recession which had the lowest GDP.
#A university town is a city which has a high percentage of university students compared to the total population of the city.
#Hypothesis: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (price_ratio=quarter_before_recession/recession_bottom)

#The following data files are available for this assignment:

#From the Zillow research data site there is housing data for the United States. In particular the datafile for all homes at a city level, City_Zhvi_AllHomes.csv, has median home sale prices at a fine grained level.
#From the Wikipedia page on college towns is a list of university towns in the United States which has been copy and pasted into the file university_towns.txt.
#From Bureau of Economic Analysis, US Department of Commerce, the GDP over time of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file gdplev.xls. For this assignment, only look at GDP data from the first quarter of 2000 onward.
#Each function in this assignment below is worth 10%, with the exception of run_ttest(), which is worth 50%.

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}

def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ],
    columns=["State", "RegionName"]  )

    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''

    utowns = pd.read_csv('university_towns.txt', sep='\n', header=None)
    snames = []
    rnames = []
    for i in range(len(utowns)):
        if '[edit]' in utowns.iloc[i][0]:
            snames.append(utowns.iloc[i][0])
        else:
            rnames.append(utowns.iloc[i][0])
            snames.append('1')
    for i in range(len(snames)):
        if '1' in snames[i]:
            snames[i] = snames[i-1].replace('[edit]','')
    for x in snames:
        if '[edit]' in x:
            snames.remove(x)
    rnames = [y.split(' (')[0] for y in rnames]
    utowns = pd.DataFrame({'State':snames, 'RegionName':rnames}, columns=['State', 'RegionName'])
    return utowns
get_list_of_university_towns()

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a
    string value in a format such as 2005q3'''
    gdplev = pd.read_excel('gdplev.xls', skiprows=220, header=None)
    gdplev = gdplev[[4,6]]
    for i in range(len(gdplev)):
        if (gdplev.iloc[i-4][6] > gdplev.iloc[i-3][6]) and (gdplev.iloc[i-3][6] > gdplev.iloc[i-2][6])\
        and (gdplev.iloc[i-2][6] < gdplev.iloc[i-1][6]) and (gdplev.iloc[i-1][6] < gdplev.iloc[i][6]):
            start_q = gdplev.iloc[(i-4)-1][4]
            #in case if the start is underestimated
            #!!! I made a mistake here, quarter_before_recession means which we need the quarter right before the start.
            #However, my comprehension was the 1st quarter begin to recess.

            #for j in range(1, i-4):    (For the 1st quarter starts to recess b4 the recession)
                #if gdplev.iloc[i-4-j][6] > gdplev.iloc[i-4-(j-1)][6]:
                    #start_q = gdplev.iloc[i-4-j][4]
                #elif gdplev.iloc[i-4-j][6] < gdplev.iloc[i-4-(j-1)][6]:
                    #break
    return start_q
get_recession_start()

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a
    string value in a format such as 2005q3'''
    gdplev = pd.read_excel('gdplev.xls', skiprows=220, header=None)
    gdplev = gdplev[[4,6]]
    for i in range(len(gdplev)):
        if (gdplev.iloc[i-4][6] > gdplev.iloc[i-3][6]) and (gdplev.iloc[i-3][6] > gdplev.iloc[i-2][6])\
        and (gdplev.iloc[i-2][6] < gdplev.iloc[i-1][6]) and (gdplev.iloc[i-1][6] < gdplev.iloc[i][6]):
            end_q = gdplev.iloc[i][4]
    return end_q
get_recession_end()

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a
    string value in a format such as 2005q3'''
    gdplev = pd.read_excel('gdplev.xls', skiprows=220, header=None)
    gdplev = gdplev[[4,6]]
    for i in range(len(gdplev)):
        if (gdplev.iloc[i-4][6] > gdplev.iloc[i-3][6]) and (gdplev.iloc[i-3][6] > gdplev.iloc[i-2][6])\
        and (gdplev.iloc[i-2][6] < gdplev.iloc[i-1][6]) and (gdplev.iloc[i-1][6] < gdplev.iloc[i][6]):
            bottom_q = gdplev.iloc[i-2][4]
    return bottom_q
get_recession_bottom()

def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].

    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.

    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''

def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].

    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.

    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''

    houses = pd.read_csv('City_Zhvi_AllHomes.csv', header=0)
    houses['State'].replace(states, inplace=True)
    houses = houses.set_index(["State","RegionName"])
    houses = houses.iloc[:,49:]
    houses = houses.groupby(pd.PeriodIndex(houses.columns, freq='Q'), axis=1).mean()
    houses.columns = houses.columns.strftime('%Yq%q')
    #This Alternative method for the fundamental explainations of groupby quarter method, is inspired by Junho Yoo(To be a Data Scientist) on google.
    #def splitq(col):
        #if col.endswith(("01", "02", "03")):
            #quarter = col[:4]+"q1"
        #elif col.endswith(("04", "05", "06")):
            #quarter = col[:4]+"q2"
        #elif col.endswith(("07", "08", "09")):
            #quarter = col[:4]+"q3"
        #else:
            #quarter = col[:4]+"q4"
        #return quarter
    #houses = houses.groupby(splitq, axis = 1).mean()
    return houses
convert_housing_data_to_quarters()

def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values,
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence.

    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''

    utowns = get_list_of_university_towns()
    start_q = get_recession_start()
    bottom_q = get_recession_bottom()
    houses = convert_housing_data_to_quarters()
    utowns['university'] = True
    prices = pd.merge(utowns, houses, how='outer', left_on=['State', 'RegionName'], right_index=True).reset_index()
    prices['university'].replace({np.NaN: False}, inplace=True)
    prices['price_ratio'] = prices[bottom_q] - prices[start_q]
    prices = prices[['university','price_ratio']]
    prices = prices.dropna().reset_index()
    utown_pratios = []
    nonutown_pratios = []
    for i in range(len(prices)):
        if prices.iloc[i][1] == True:
            utown_pratios.append(prices.iloc[i][2])
        else:
            nonutown_pratios.append(prices.iloc[i][2])
    t, p = ttest_ind(utown_pratios, nonutown_pratios)
    better = None
    if p < 0.01:
        different = True
    else:
        different = False
    if np.mean(utown_pratios) > np.mean(nonutown_pratios):
        better = "university town"
    elif np.mean(utown_pratios) < np.mean(nonutown_pratios):
        better = "non-university town"

    return (different, p, better)
run_ttest()
