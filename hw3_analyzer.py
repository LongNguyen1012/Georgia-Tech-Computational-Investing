# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 00:04:16 2016

@author: Long Nguyen
"""
import QSTK.qstkutil.tsutil as tsu
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from DataPrepare import prepare_data

def analyze(df_value):
    
    from DataPrepare import prepare_data    
    
    # Getting the numpy ndarray of close prices.
    na_value = df_value.values
    
    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_value = na_value / na_value[0, :]
    
    # Copy the normalized prices to a new ndarry to find returns.
    na_rets = na_normalized_value.copy()
    
    # Calculate the daily returns of the prices. (Inplace calculation)
    # returnize0 works on ndarray and not dataframes.
    tsu.returnize0(na_rets)
    
    na_port_total = np.cumprod(na_rets + 1)
    daily_port_ret = np.average(na_rets)
    port_cum_ret = na_port_total[-1]
    port_vol = np.std(na_rets)
    port_sharpe = (daily_port_ret/port_vol)*np.sqrt(252)
    ls_date = df_value.index.tolist()
    dt_start = ls_date[0]
    dt_end = ls_date[-1]
    
    ls_keys = ["close"]
    ls_symbols = ["$SPX"]
    d_data = prepare_data(dt_start, dt_end, ls_keys, ls_symbols)        
        
    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values
    
    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_price = na_price / na_price[0, :]
    
    # Copy the normalized prices to a new ndarry to find returns.
    na_market_rets = na_normalized_price.copy()
    
    # Calculate the daily returns of the prices. (Inplace calculation)
    # returnize0 works on ndarray and not dataframes.
    tsu.returnize0(na_market_rets)
    
    na_market_total = np.cumprod(na_market_rets + 1)
    daily_market_ret = np.average(na_market_rets)
    market_cum_ret = na_market_total[-1]
    market_vol = np.std(na_market_rets)
    market_sharpe = (daily_market_ret/market_vol)*np.sqrt(252)
    
    
    print "The final value of the portfolio using the sample file is", dt_end, " ", df_value['Portfolio_value'][-1]
    print ""
    print "Details of the Performance of the portfolio :"
    print ""
    print "Data Range :",  dt_start, " to ", dt_end
    print ""
    print "Sharpe Ratio of Fund :", port_sharpe
    print "Sharpe Ratio of $SPX :", market_sharpe
    print ""
    print "Total Return of Fund :", port_cum_ret
    print "Total Return of $SPX :", market_cum_ret
    print ""
    print "Standard Deviation of Fund :", port_vol
    print "Standard Deviation of $SPX :", market_vol
    print ""
    print "Average Daily Return of Fund :", daily_port_ret
    print "Average Daily Return of $SPX :", daily_market_ret
    
    ls_symbols = ["Portfolio", "SP500"]
    na_port_total = np.reshape(na_port_total,(len(na_port_total),1))
    na_final = np.hstack((na_port_total, na_normalized_price))
    ldt_timestamps = df_value.index.tolist()
    plt.clf()
    plt.plot(ldt_timestamps, na_final)
    plt.legend(ls_symbols)
    plt.ylabel('Portfolio Normalized Value')
    plt.xlabel('Date')
    plt.savefig('simulated_portfolio.pdf', format='pdf')
    
    
def BollingerBands(d_data, period = 20, plot=True):
    
    import talib as ta
    import pandas as pd
    import copy

    # Creating an empty dataframe
    df_close = d_data['close']
    ls_symbols = df_close.columns.values.tolist()
    ldt_timestamps = df_close.index

    df_bollinger = copy.deepcopy(df_close)
    df_bollinger = df_bollinger * np.NAN 
    
    for s_sym in ls_symbols:
        na_price = df_close[s_sym].values
        # TA-Lib takes on 1-D array
        
        up,_,bottom = ta.BBANDS(na_price, period, 1, 1)
        na_mva = pd.rolling_mean(na_price, period)
        na_mvstd = pd.rolling_std(na_price, period)
        f_bollinger = (na_price - na_mva) / na_mvstd
        
        up = np.reshape(up, (len(up), 1))
        bottom = np.reshape(bottom, (len(bottom), 1))
        na_price = np.reshape(na_price, (len(na_price), 1))
        
        na_final = np.hstack((up, bottom, na_price))
    
        if plot == True:
            f, ax = plt.subplots(2, sharex=True)
            
            ax[0].plot(ldt_timestamps, na_final)
            ax[0].legend(['up','bottom', 'price'])
            
            ax[1].plot(ldt_timestamps, f_bollinger)
            ax[1].axhline(y=1, color='b')
            ax[1].axhline(y=-1, color='b')
            ax[1].vlines(x = ldt_timestamps[np.where(f_bollinger > 1)], ymin = -3,  ymax = 4, color = 'r')
            ax[1].vlines(x = ldt_timestamps[np.where(f_bollinger < -1)], ymin = -3,  ymax = 4, color = 'green')
            ax[1].fill_between(ldt_timestamps, 1, -1, facecolor = 'blue', alpha = 0.1)
    
            plt.title(s_sym)
    
        df_bollinger[s_sym] = f_bollinger
        
    return df_bollinger
    

if __name__ == '__main__':
    
    # List of symbols
    ls_symbols = ['MSFT','GOOG']  
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    
    # Start and End date of the charts
    dt_start = dt.datetime(2010, 1, 1)
    dt_end = dt.datetime(2010, 12, 31)   
    
    d_data = prepare_data(dt_start, dt_end, ls_keys, ls_symbols)
    df_bollinger = BollingerBands(d_data)
    
    