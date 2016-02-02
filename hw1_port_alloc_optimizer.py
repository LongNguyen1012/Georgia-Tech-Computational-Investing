'''
(c) 2011, 2012 Georgia Tech Research Corporation
This source code is released under the New BSD license.  Please see
http://wiki.quantsoftware.org/index.php?title=QSTK_License
for license details.

Created on January, 24, 2013

@author: Sourabh Bajaj
@contact: sourabhbajaj@gatech.edu
@summary: Example tutorial code.
'''

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools


def drange(start, stop, step):
     r = start
     while r < stop:
         yield r
         r = round((step + r),1)


def prepare_data(dt_start, dt_end, ls_keys, ls_symbols): 
    ''' Get data, fill na and normalize price data'''
    
    import QSTK.qstkutil.DataAccess as da
    
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)
    
    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
    
    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo', cachestalltime=0)
    
    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    
    # Filling the data for NAN
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
        
    return d_data
    
    
def optimizer(dt_start, dt_end, ls_symbols):
    ''' Main Function''' 
    
    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    
    # Read in and prepare data    
    d_data = prepare_data(dt_start, dt_end, ls_keys, ls_symbols)
    
    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values
    
    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_price = na_price / na_price[0, :]
    
    # Copy the normalized prices to a new ndarry to find returns.
    na_rets = na_normalized_price.copy()
    
    # Calculate the daily returns of the prices. (Inplace calculation)
    # returnize0 works on ndarray and not dataframes.
    tsu.returnize0(na_rets)
    
    # Optimizing
    ls_sharpe = []
    pos_combin = np.asarray(list(itertools.product(list(drange(0,1.1,0.1)),repeat = len(ls_symbols))))
    sum_value = np.sum(pos_combin, axis = 1)
    legal_combin = pos_combin[np.where(sum_value == 1.0)] 
    for combin in legal_combin:
        # Estimate portfolio returns
        lf_port_alloc = list(combin)
        na_portrets = np.sum(na_rets * lf_port_alloc, axis=1)
        daily_ret = np.average(na_portrets)
        vol = np.std(na_portrets)
        sharpe = (daily_ret/vol)*np.sqrt(252)
        ls_sharpe.append(sharpe)
        
    ls_sharpe = np.asarray(ls_sharpe)
    best_combin = legal_combin[np.argmax(ls_sharpe)]
    
    lf_port_alloc = best_combin
    na_portrets = np.sum(na_rets * lf_port_alloc, axis=1)
    na_port_total = np.cumprod(na_portrets + 1)
    daily_ret = np.average(na_portrets)
    cum_ret = na_port_total[-1]
    vol = np.std(na_portrets)
    sharpe = (daily_ret/vol)*np.sqrt(252)
    
    ls_keys = ["close"]
    ls_symbols = ["$SPX"]
    d_data = prepare_data(dt_start, dt_end, ls_keys, ls_symbols)        
        
    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values
    
    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_price = na_price / na_price[0, :]
    
    # Plotting the prices with x-axis=timestamps
    dt_timeofday = dt.timedelta(hours=16)
    
    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
    
    ls_symbols = ["Portfolio", "SP500"]
    na_port_total = np.reshape(na_port_total,(len(na_port_total),1))
    na_final = np.hstack((na_port_total, na_normalized_price))
    plt.clf()
    plt.plot(ldt_timestamps, na_final)
    plt.legend(ls_symbols)
    plt.ylabel('Portfolio Normalized Value')
    plt.xlabel('Date')
    plt.savefig('optimized_portfolio.pdf', format='pdf')
    
    return best_combin, vol, cum_ret, daily_ret, sharpe

if __name__ == '__main__':
    
    # List of symbols
    ls_symbols = ['C', 'GS', 'IBM', 'HNZ']  
    
    # Start and End date of the charts
    dt_start = dt.datetime(2010, 1, 1)
    dt_end = dt.datetime(2010, 12, 31)   
    
    best_combin, vol, cum_ret, daily_ret, sharpe = optimizer(dt_start, dt_end, ls_symbols)
