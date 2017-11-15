# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:27:29 2017

@author: Akhilesh
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def bytedate2num(fmt):
    def converter(b):
        return mdates.strpdate2num(fmt)(b.decode('ascii'))
    return converter

def graphRawFX():
    date, bid, ask = np.loadtxt('GBPUSD/GBPUSD1d.txt', 
                                unpack=True,
                                delimiter=',',
                                converters={0:bytedate2num('%Y%m%d%H%M%S')})
    plt.figure(figsize=(10,7))
    ax1 = plt.subplot2grid((40,40), (0,0), rowspan=40, colspan=40)
    ax1.plot(date,bid,color = 'b')
    ax1.plot(date,ask,color = 'g')
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    
    ax1_2 = ax1.twinx()
    ax1_2.fill_between(date, 0, (ask-bid), facecolor='g', alpha=.3)
    
    plt.subplots_adjust(bottom=.23)
    
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    graphRawFX()