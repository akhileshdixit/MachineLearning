# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:11:43 2017

@author: Akhilesh
"""

import datetime as dt
from matplotlib import style
import pandas_datareader.data as web

style.use('ggplot')

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2016, 12, 31)

df = web.DataReader('TSLA', 'yahoo', start, end)
print(df.tail())
