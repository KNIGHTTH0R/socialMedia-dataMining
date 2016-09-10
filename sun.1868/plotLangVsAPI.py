import numpy as np
import scipy
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
import sys, csv
import ast
#open and read the data file 
f = open("APIandLangID.txt", "r") 
lines = f.readlines()
language = []
number = []
for x in lines:
    x = x.strip('\r\n')
    language.append(x.split(',')[0])
df = pd.read_csv("APIandLangID.txt", converters={"colA": ast.literal_eval })
f.close()
#preparation for drawing the picture
languageNew = []
languageNew  = [language[i] for i in range(1, len(language))]
for i in range(len(languageNew)):
    number.append(df.colA.iloc[i])
#drawing the picture
n_bins = len(languageNew)
x = np.asarray(number)
fig = plt.figure()
ax = fig.add_subplot(111)
yTickMarks = languageNew
ytickNames = ax.set_yticklabels(yTickMarks)
plt.setp(ytickNames, rotation=45, fontsize=10)
label1 = "API"
label2 = "langid.py"
ax.hist(x, n_bins, normed=1, histtype='bar', stacked=True ,orientation = 'horizontal',label=[label1, label2])
ax.legend()
fig.suptitle('stacked bar of percentage of different language checked by API and langid.py')
ax.xaxis.tick_top()
plt.show()
