'''
(c) 2011, 2012 Georgia Tech Research Corporation
This source code is released under the New BSD license.  Please see
http://wiki.quantsoftware.org/index.php?title=QSTK_License
for license details.

Created on January, 23, 2013

@author: Sourabh Bajaj
@contact: sourabhbajaj@gatech.edu
@summary: Event Profiler Tutorial
'''


import numpy as np
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import new_event_profiler as ep
from hw3_analyzer import BollingerBands

"""
Accepts a list of symbols along with start and end date
Returns the Event Matrix which is a pandas Datamatrix
Event matrix has the following structure :
    |IBM |GOOG|XOM |MSFT| GS | JP |
(d1)|nan |nan | 1  |nan |nan | 1  |
(d2)|nan | 1  |nan |nan |nan |nan |
(d3)| 1  |nan | 1  |nan | 1  |nan |
(d4)|nan |  1 |nan | 1  |nan |nan |
...................................
...................................
Also, d1 = start date
nan = no information about any event.
1 = status bit(positively confirms the event occurence)
"""


def find_events(ls_symbols, d_data, market = "SPY"):
    ''' Finding the event dataframe '''
    df_close = d_data['close']
    ts_market = df_close.pop(market)

    print "Finding Events"

    # Creating an empty order list
    ls_orders = []

    # Time stamps for the event range
    ldt_timestamps = df_close.index

    # Create Bollinger Bands Data
    df_bollinger = BollingerBands(d_data, plot=False)
    
    for s_sym in ls_symbols:
        for i in range(1, len(ldt_timestamps)):
            # Calculating the returns for this timestamp
            f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
            f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]
            f_boll_today = df_bollinger[s_sym].ix[ldt_timestamps[i]]
            f_boll_yest = df_bollinger[s_sym].ix[ldt_timestamps[i - 1]]
                        
            
            # Event is found if the symbol suddenly drops below $5.00
            if f_symprice_yest >= 6.0 and f_symprice_today < 6.0:
                buy_date = ldt_timestamps[i].to_datetime()
                if (i + 5) < len(ldt_timestamps):
                    sell_date = ldt_timestamps[i + 5].to_datetime()
                else:
                    sell_date = ldt_timestamps[-1].to_datetime()
                ls_buy_order = [buy_date.year, buy_date.month, buy_date.day, s_sym, "Buy", 100]
                ls_sell_order = [sell_date.year, sell_date.month, sell_date.day, s_sym, "Sell", 100]
                ls_orders.append(ls_buy_order) 
                ls_orders.append(ls_sell_order)           
            
    ls_orders = np.asarray(ls_orders)
    
    return ls_orders


def write_output(filename, ls_orders):
    import csv
    writer = csv.writer(open(filename, 'wb'), delimiter = ",")
    for row in ls_orders:
        writer.writerow(row)
        

if __name__ == '__main__':
    dt_start = dt.datetime(2008, 1, 1)
    dt_end = dt.datetime(2009, 12, 31)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

    dataobj = da.DataAccess('Yahoo')
    ls_symbols = dataobj.get_symbols_from_list('sp5002012')
    ls_symbols.append('SPY')

    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    df_events, ls_orders = find_events(ls_symbols, d_data)
    print "Creating Study"
    ep.eventprofiler(df_events, d_data, i_lookback=20, i_lookforward=20,
                s_filename='MyEventStudy.pdf', b_market_neutral=False, b_errorbars=True, use_actual_close=True,
                s_market_sym='SPY')
    
    write_output("test_orders.csv", ls_orders)
