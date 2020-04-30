import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')
df.head()
#We need to sift out the years&dates, groupby TMAX&TMIN from the sample of dataframe.

years = list(range(len(df)))
days = list(range(len(df)))
for i in range(len(df)):
    years[i] = (df['Date'][i]).split('-')[0]
    days[i] = (df['Date'][i]).split('-')[1] + (df['Date'][i]).split('-')[2]
df['years'] = years
df['days'] = days
df = df[df['days'] != '0229']
df15 = df[df['years'] == '2015']
df_rest = df[~(df['years'] == '2015')]
df15max = df15[df15['Element'] == 'TMAX']
df15min = df15[df15['Element'] == 'TMIN']
df_restmax = df_rest[df_rest['Element'] == 'TMAX']
df_restmin = df_rest[df_rest['Element'] == 'TMIN']
max15 = df15max.groupby('days').agg({'Data_Value':max}).reset_index()
min15 = df15min.groupby('days').agg({'Data_Value':min}).reset_index()
max_rest = df_restmax.groupby('days').agg({'Data_Value':max}).reset_index()
min_rest = df_restmin.groupby('days').agg({'Data_Value':min}).reset_index()
maxim15 = pd.merge(max15, max_rest, on = 'days', how = 'outer')
minum15 = pd.merge(min15, min_rest, on = 'days', how = 'outer')
# Data_Value_x is the max15, Data_Value_y is the max_rest, comparying for the break
breakmax = maxim15[maxim15['Data_Value_x'] > maxim15['Data_Value_y']].drop(['Data_Value_y'], axis=1)
breakmin = minum15[minum15['Data_Value_x'] < minum15['Data_Value_y']].drop(['Data_Value_y'], axis=1)
# Check if we only get the numbers of the breakvalues for the max_2015, which are stored in Data_Value_x
print(breakmax)

# Lets start the plot by %matplotlib inline, fit-in plot without plt.show()
%matplotlib inline
plt.figure(figsize=(14,9))

plt.plot(max_rest.index, max_rest['Data_Value'], c = 'blue', label ='Ten year record high')
plt.plot(min_rest.index, min_rest['Data_Value'], c = 'green', label ='Ten year record low')

# To be honest, I do not wannt to seperate the colors for break, initial set was all red
# However, the legend has to tell the difference, basically the top is the max break records, bottom reversely
plt.scatter(breakmax.index, breakmax['Data_Value_x'], c = 'red', label = "2015 broken record high")
plt.scatter(breakmin.index, breakmin['Data_Value_x'], c = 'brown', label = "2015 broken record low")

plt.gca().fill_between(range(len(max_rest)),
                       np.array(max_rest['Data_Value']),
                       np.array(min_rest['Data_Value']),
                       facecolor='orange',
                       alpha=0.15)

plt.xlabel('365-days w/o 2-29', fontsize=16)
plt.ylabel('Records of Temperature', fontsize=16)
plt.title('Ten-year-record(05-14) with broken scatter-records in Michigan, United States 2015', fontsize=20)
plt.legend(loc = 2, frameon = False, fontsize = 11)
#plt.xticks( np.linspace(15,15 + 30*11 , num = 12), (r'Jan', r'Fev', r'Mar', r'Abr', r'Mai', r'Jun', r'Jul', r'Ago', r'Set', r'Out', r'Nov', r'Dec') )

# Alternative: Create a download link for the file in this project
# file ='data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv'
# !cp "$file" .
# from IPython.display import HTML
# link = '<a href="{0}" download>Click here to download {0}</a>'
# HTML(link.format(file.split('/')[-1]))
