# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:31:23 2017

@author: Akhilesh
Handling Data and Graphics

"""

#import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
#import pandas_datareader.data as web

style.use('ggplot')

#start = dt.datetime(2000, 1, 1)
#end = dt.datetime(2016, 12, 31)

#dfw = web.DataReader('TSLA', 'yahoo', start, end)
#dfw.to_csv('tsla.csv')

dfr = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

#print(dfr.head())

print(dfr[['Open', 'High']].head())

dfr['Adj Close'].plot()
plt.show()