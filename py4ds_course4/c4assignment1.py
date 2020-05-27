import pandas as pd

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df.head(10)

'''I have wasted a lot of time on wrtting the loops and trying to handle with year, month, day seperately.
    however, it failed with the "value is trying to be set on a copy of a slice from a DataFrame" errors.
    here is my code of original and I am studing on how to evade this kind of errors, just a copy and ideas kept.'''

def date_sorter():
    '''04/20/2009; 04/20/09; 4/20/09; 4/3/09
    Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
    20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
    Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
    Feb 2009; Sep 2009; Oct 2010
    6/2008; 12/2009
    2009; 2010'''

    # Your code here

    month = {'Jan': '01', 'Feb': '02' ,'Mar': '03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08',
            'Sep':'09','Oct':'10','Nov':'11','Dec':'12'}

    r1 = r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})'
    r2 = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z]*[.]?[- ](\d{1,2})[,]?[- ](\d{4})'
    r3 = r'(\d{1,2}) ((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z]*[.,]? (\d{4})'
    r4 = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z]* (\d{1,2})[a-z]{2}[,] (\d{4})'
    r5 = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z]* (\d{4})'
    r6 = r'(\d{1,2})[/](\d{4})'
    r7 = r'([1|2]\d{3})'

    r = '(%s|%s|%s|%s|%s|%s|%s)' %(r1, r2, r3, r4, r5, r6, r7)
    date = df.str.extract(r)
    date = date.fillna(0)
    #day index 2, 5, 7, 11
    #month index 1, 4, 8, 10, 13, 15
    #year index 3, 6, 9, 12, 14, 16, 17
    for i in range(len(date)):
        if date.iloc[i][1] == 0:
            if date.iloc[i][4] != 0:
                date.iloc[i][1]=date.iloc[i][4]
            elif date.iloc[i][8] != 0:
                date.iloc[i][1]=date.iloc[i][8]
            elif date.iloc[i][10] != 0:
                date.iloc[i][1]=date.iloc[i][10]
            elif date.iloc[i][13] != 0:
                date.iloc[i][1]=date.iloc[i][13]
            elif date.iloc[i][15] != 0:
                date.iloc[i][1]=date.iloc[i][15]
        if date.iloc[i][2] == 0:
            if date.iloc[i][5] != 0:
                date.iloc[i][2]=date.iloc[i][5]
            elif date.iloc[i][7] != 0:
                date.iloc[i][2]=date.iloc[i][7]
            elif date.iloc[i][11] != 0:
                date.iloc[i][2]=date.iloc[i][11]
        if date.iloc[i][3] == 0:
            if date.iloc[i][6] != 0:
                date.iloc[i][3]=date.iloc[i][6]
            elif date.iloc[i][9] != 0:
                date.iloc[i][3]=date.iloc[i][9]
            elif date.iloc[i][12] != 0:
                date.iloc[i][3]=date.iloc[i][12]
            elif date.iloc[i][14] != 0:
                date.iloc[i][3]=date.iloc[i][14]
            elif date.iloc[i][16] != 0:
                date.iloc[i][3]=date.iloc[i][16]
            elif date.iloc[i][17] != 0:
                date.iloc[i][3]=date.iloc[i][17]


    return date



###eaiest way
def date_sorter():
    '''04/20/2009; 04/20/09; 4/20/09; 4/3/09
    Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
    20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
    Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
    Feb 2009; Sep 2009; Oct 2010
    6/2008; 12/2009
    2009; 2010'''

    # Your code here

    month = {'Jan': '01', 'Feb': '02' ,'Mar': '03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08',
            'Sep':'09','Oct':'10','Nov':'11','Dec':'12'}

    #code
    r1 = r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})'
    r2 = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z]*[.]?[- ](\d{1,2})[,]?[- ](\d{4})'
    r3 = r'(\d{1,2}) ((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z]*[.,]? (\d{4})'
    r4 = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z]* (\d{1,2})[a-z]{2}[,] (\d{4})'
    r5 = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z]* (\d{4})'
    r6 = r'(\d{1,2})[/](\d{4})'
    r7 = r'([1|2]\d{3})'

    r = '(%s|%s|%s|%s|%s|%s|%s)' %(r1, r2, r3, r4, r5, r6, r7)
    date = df.str.extract(r)
    date = date[0]

    #when I excute pd.to_datetime, I get errors with unknown string format, so I checked string types
    #and I find out that Janaury and Decemeber are typos
    #date.unique()

    date = date.str.replace('Janaury', 'January')
    date = date.str.replace('Decemeber', 'December')

    #pandas to date will automatically change year w/o day&month to year-01-01, and w/o date to year-month-01
    #I have written a lot of loops that most of my time got wasted,and I find out that this maybe the easiest solution
    #to this problem when you use pandas datetime, inspired by agniiyer github from google

    date = pd.Series(pd.to_datetime(date))
    date = date.sort_values(ascending=True)
    rank = pd.Series(date.index)

    return rank
