import numpy as np
import matplotlib.pyplot as plt
from numpy import array
import matplotlib.patches as mpatches
f=open("file.txt","r")
lines=f.readlines()
language=[]
number =[]
for x in lines:
    x = x.strip('\r\n')
    language.append(x.split(',')[0])
    number.append(x.split(',')[1])
number  = [ float(i) for i in number ]
f.close()
x = list(range(20))
y = number

fig, ax = plt.subplots()
ax.plot(x, y, 'bo-')

for (X, Y, Z) in zip(x, y, language):
    ax.annotate(Z, xy=(X,Y), xytext=(-10, 10), ha='right', textcoords='offset points', arrowprops=dict(arrowstyle='->', shrinkA=0))

fig.suptitle('Percentage of different languages in USA', fontsize=20)
plt.xlabel('Ranking of languages', fontsize=16)
plt.ylabel('Percentage', fontsize=16)

r = x[0:9]
a = plt.axes([0.2, 0.6, .2, .2], axisbg='y')
plt.scatter(language[:len(r)], y[0:9])
red_patch = mpatches.Patch(color='b', label='Top 10')
plt.legend(handles=[red_patch])

plt.show()
