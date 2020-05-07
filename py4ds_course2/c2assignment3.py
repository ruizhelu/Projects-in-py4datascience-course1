# Use the following data for this assignment:

import pandas as pd
import numpy as np

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650),
                   np.random.normal(43000,100000,3650),
                   np.random.normal(43500,140000,3650),
                   np.random.normal(48000,70000,3650)],
                  index=[1992,1993,1994,1995])
# get the transpose of dataframe to check all the values descriptions we need
df.T.describe()

mean = list(df.mean(axis=1))
std = list(df.std(axis=1))
# get the y-error, z-values for us
yerr = list((1.96*(x))/(np.sqrt(3650)) for x in std)
min_y = list(mean[i]-yerr[i] for i in range(len(mean)))
max_y = list(mean[i]+yerr[i] for i in range(len(mean)))
range_y = list(max_y[i]-min_y[i] for i in range(len(mean)))
print(range_y)
# we need the range of confidence interval to set down the colors
# i have not set the test mechanic for y input, try to put a valid float number
y = float(input("Put your y-value of interest in(must be a valid float number!): "))

percent = list((y-min_y[i])/range_y[i] for i in range(len(mean)))
print(percent)
color = []
for x in percent:
    if x <= 0.09:
        color.append("darkred")
    if (0.09 < x) & (x <= 0.18):
        color.append("red")
    if (0.18 < x) & (x <= 0.27):
        color.append("chocolate")
    if (0.27 < x) & (x <= 0.36):
        color.append("sandybrown")
    if (0.36 < x) & (x <= 0.45):
        color.append("peachpuff")
    if (0.45 < x) & (x <= 0.55):
        color.append("white")
    if (0.55 < x) & (x <= 0.64):
        color.append("lightblue")
    if (0.64 < x) & (x <= 0.73):
        color.append("lightskyblue")
    if (0.73 < x) & (x <= 0.82):
        color.append("cornflowerblue")
    if (0.82 < x) & (x <= 0.91):
        color.append("royalblue")
    if x > 0.91:
        color.append("navy")
# we store 4 colors we need into a list for barplot
print(color)

import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline

x = (1992,1993,1994,1995)
y_pos = range(len(x))

plt.figure(figsize=(9, 9))
plt.bar(y_pos , height=mean, yerr=yerr, width=1, edgecolor='black', error_kw={'capsize': 20, 'elinewidth': 2}, color=color)
plt.axhline(y=y, color='grey')
plt.annotate('y = {}'.format(y), [0,y])
plt.xticks(y_pos, x)
plt.tick_params(bottom='off')

fig, ax = plt.subplots(figsize=(7, 1))
fig.subplots_adjust(bottom=0.75)

cmap = mpl.colors.ListedColormap(['navy', 'royalblue', 'cornflowerblue', 'lightskyblue', 'lightblue', 'white',
                                  'peachpuff', 'sandybrown', 'chocolate', 'red', 'darkred'])
bounds = [0.00, 0.09, 0.18, 0.27, 0.36, 0.45, 0.55, 0.64, 0.73, 0.82, 0.91, 1.00]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb3 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                boundaries=bounds,
                                ticks=bounds,
                                spacing='uniform',
                                orientation='horizontal')
