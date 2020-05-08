#https://en.wikipedia.org/wiki/List_of_Dallas_Mavericks_seasons
#https://en.wikipedia.org/wiki/List_of_San_Antonio_Spurs_seasons
#https://en.wikipedia.org/wiki/List_of_Houston_Rockets_seasons

from fake_useragent import UserAgent
import requests
from bs4 import BeautifulSoup
# this parse html method introduced from
# "https://www.ankuroh.com/programming/automation/web-scraping-with-python-text-scraping-wikipedia/"
# Very easy to understand and useful.

# prevent from blocked by the web
ua1 = UserAgent()
randomHeader = {'User-Agent':str(ua1.random)}

scrapeLink = 'https://en.wikipedia.org/wiki/List_of_Dallas_Mavericks_seasons'
page = requests.get(scrapeLink, randomHeader)


soup = BeautifulSoup(page.content, 'html.parser')
# Find the second table we need, with its years, win%
table = soup.find_all('table')[2]

print(table)

# row values list mining all the values we need(eg, win%), except for seasons
rowValList = []

for i in range(len(table.find_all('td'))):
    rowVal = table.find_all('td')[i].get_text()
    rowValList.append(rowVal)
print(rowValList)

# hence we have to parse another list for the seasons
SeasonList = []

for i in range(len(table.find_all('th'))):
    season_val = table.find_all('th')[i].get_text()
    SeasonList.append(season_val)
print(SeasonList)

import pandas as pd

winrList = []
for i in range(7, len(rowValList), 12):
    winrList.append(rowValList[i].replace('\n',''))

seasonList = []
for i in range(13, len(SeasonList)):
    seasonList.append(SeasonList[i].replace('\n',''))

df1 = pd.DataFrame()
df1['Seasons'] = seasonList
df1['Win_Rate'] = winrList
# There are still two seasons with specific symbols [c]/[d], but I would pass it
df1.head()

# Do the same thing for San Antonio Spurs
ua2 = UserAgent()
randomHeader2 = {'User-Agent':str(ua2.random)}

scrapeLink2 = 'https://en.wikipedia.org/wiki/List_of_San_Antonio_Spurs_seasons'
page2 = requests.get(scrapeLink2, randomHeader2)


soup2 = BeautifulSoup(page2.content, 'html.parser')
# Find the second table we need, with its years, win%
table2 = soup2.find_all('table')[2]

rowValList2 = []

for i in range(len(table2.find_all('td'))):
    rowVal2 = table2.find_all('td')[i].get_text()
    rowValList2.append(rowVal2)

# hence we have to parse another list for the seasons
SeasonList2 = []

for i in range(len(table.find_all('th'))):
    season_val2 = table.find_all('th')[i].get_text()
    SeasonList2.append(season_val2)
#print(rowValList2)

winrList2 = []
for i in range(len(rowValList2)):
    if '.' in rowValList2[i]:
        winrList2.append(rowValList2[i].replace('\n',''))
# Remove the first 13 elements in ABA record and two extra terms
winrList2 = winrList2[13:]
winrList2.remove('Gregg Popovich (CoY)Kawhi Leonard (FMVP)R.C. Buford (EoY)')
winrList2.remove('Kawhi Leonard (DPoY)R.C. Buford (EoY)')
#print(len(winrList2))

seasonList2 = []
for i in range(13, len(SeasonList2)):
    seasonList2.append(SeasonList2[i].replace('\n',''))
#print(seasonList2)

df2 = pd.DataFrame()
df2['Seasons'] = seasonList2
df2['Win_Rate'] = winrList2
# There are still two seasons with specific symbols [c]/[d], but I would pass it
df2.head()

# Do the same thing for Houston Rockets
ua3 = UserAgent()
randomHeader3 = {'User-Agent':str(ua3.random)}

scrapeLink3 = 'https://en.wikipedia.org/wiki/List_of_Houston_Rockets_seasons'
page3 = requests.get(scrapeLink3, randomHeader3)


soup3 = BeautifulSoup(page3.content, 'html.parser')
# Find the second table we need, with its years, win%
table3 = soup3.find_all('table')[2]

rowValList3 = []

for i in range(len(table3.find_all('td'))):
    rowVal3 = table3.find_all('td')[i].get_text()
    rowValList3.append(rowVal3)

# hence we have to parse another list for the seasons
SeasonList3 = []

for i in range(len(table.find_all('th'))):
    season_val3 = table.find_all('th')[i].get_text()
    SeasonList3.append(season_val3)
#print(rowValList2)

winrList3 = []
for i in range(len(rowValList3)):
    if '.' in rowValList3[i]:
        winrList3.append(rowValList3[i].replace('\n',''))
# Remove the first 13 elements in ABA record and two extra terms
winrList3 = winrList3[13:]
#print(len(winrList3))

seasonList3 = []
for i in range(13, len(SeasonList3)):
    seasonList3.append(SeasonList3[i].replace('\n',''))
#print(len(seasonList3))

df3 = pd.DataFrame()
df3['Seasons'] = seasonList3
df3['Win_Rate'] = winrList3
# There are still two seasons with specific symbols [c]/[d], but I would pass it
df3.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib notebook

# use the 'seaborn-colorblind' style
plt.style.use('seaborn-paper')

# Hence we have parsed and mined three tables df1, df2 and df3 for Mavericks, Spurs and Rockets
df = pd.merge(df1, df2, how='outer', left_on=['Seasons'], right_on=['Seasons'])
df = pd.merge(df, df3, how='outer', left_on=['Seasons'], right_on=['Seasons'])
df = df.rename(columns={"Win_Rate_x": "Dallas Mavericks", "Win_Rate_y": "San Antonio Spurs", "Win_Rate": "Houston Rockets" })
x = df['Seasons']
y1 = df['Dallas Mavericks']
y2 = df['San Antonio Spurs']
y3 = df['Houston Rockets']
df = df.set_index('Seasons')
df = df.astype(float)
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
ax = df.plot.kde(ax=ax, title='Kernel-Density distributions of Texas teams Win-Rate%');
ax1 = df.plot.box(ax=ax1, title='Boxes of Texas teams Win-Rate%');

ax2 = df.plot(subplots=True, figsize=(9,9), title='Comparison of Texas teams Win-Rate%')
