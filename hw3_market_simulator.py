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

# Third Party Imports
import datetime as dt
import pandas as pd
import numpy as np
import hw3_analyzer as analyzer

def read_data(filename):
    import csv
    ls_date = []
    ls_symbols = []
    with open(filename, "rU") as data_file:
        reader = csv.reader(data_file, delimiter = ",")
        for row in reader:
            year = int(row[0])
            month = int(row[1])
            day = int(row[2])
            ls_date.append(dt.datetime(year, month, day, 16, 0, 0))
            ls_symbols.append(row[3])
    
    ls_date.sort()  
    dt_start = ls_date[0]
    dt_end = ls_date[-1]
    dt_timeofday = dt.timedelta(hours=16)
    ls_date = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
    ls_symbols = list(set(ls_symbols))
    
    return ls_date, ls_symbols


def create_trade_matrix(ls_date, ls_symbols):
    zero_data = np.zeros(shape=(len(ls_date),len(ls_symbols)))
    df_trade = pd.DataFrame(zero_data, columns=ls_symbols, index=ls_date)

    return df_trade
    
    
def update_trade_matrix(df_trade, filename):
    import csv
    with open(filename, "rU") as data_file:
        reader = csv.reader(data_file, delimiter = ",")
        for row in reader:
            year = int(row[0])
            month = int(row[1])
            day = int(row[2])
            dt_date = dt.datetime(year, month, day, 16, 0, 0)
            s_symbol = row[3]
            b_action = row[4] # Buy or sell
            i_share = int(row[5])
            if b_action == "Buy":
                df_trade[s_symbol].ix[dt_date] += i_share
            else:
                df_trade[s_symbol].ix[dt_date] += -i_share

    return df_trade
    
    
def create_cash_matrix(ls_date, begin_cash):
    zero_data = np.zeros(shape=(len(ls_date),1))
    df_cash = pd.DataFrame(zero_data, columns=["_CASH"], index=ls_date)
    df_cash["_CASH"].iloc[0] = begin_cash
    
    return df_cash
    

def convert_trade_matrix_to_holding_matrix(df_trade):
    ldt_timestamps = df_trade.index.tolist()
    ls_symbols = df_trade.columns.values.tolist()
    zero_data = np.zeros(shape=(len(ldt_timestamps),len(ls_symbols)))
    df_holding = pd.DataFrame(zero_data, columns=ls_symbols, index=ldt_timestamps)
    for s_symbol in ls_symbols:
        df_holding[s_symbol] = np.cumsum(df_trade[s_symbol])

    return df_holding
    
    
def update_cash_matrix(df_cash, df_trade):
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ls_date = df_trade.index.tolist()
    ls_symbols = df_trade.columns.values.tolist()
    dt_start = ls_date[0]
    dt_end = ls_date[-1]
    d_data = prepare_data(dt_start, dt_end, ls_keys, ls_symbols)
    df_price = d_data['close']
    for i in range(len(ls_date)):
        for s_symbol in ls_symbols:
            i_trade = df_trade[s_symbol].ix[ls_date[i]]
            f_price = df_price[s_symbol].ix[ls_date[i]]
            df_cash["_CASH"].ix[ls_date[i]] += (-i_trade) * f_price
    df_cash = np.cumsum(df_cash)
    
    return df_cash    
    

def port_value(df_cash, df_holding):
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ls_date = df_holding.index.tolist()
    ls_symbols = df_holding.columns.values.tolist()
    dt_start = ls_date[0]
    dt_end = ls_date[-1]
    d_data = prepare_data(dt_start, dt_end, ls_keys, ls_symbols)
    na_price = d_data['close'].values
    na_holding = df_holding.values
    na_cash = df_cash.values
    na_port_value = np.sum(na_price * na_holding, axis = 1)
    na_port_value = np.reshape(na_port_value, (len(na_port_value), 1))
    na_port_value = na_port_value[:,0] + na_cash[:,0]
    df_value = pd.DataFrame(na_port_value, columns = ['Portfolio_value'], index = ls_date)    
    
    return df_value

    
def write_output(filename, df_value):
    import csv
    writer = csv.writer(open(filename, 'wb'), delimiter = ",")
    for row_index in df_value.index:
        row_to_enter = [row_index, df_value['Portfolio_value'][row_index]]
        writer.writerow(row_to_enter)
    
    
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


def marketsim(begin_cash, orders_file, output_file):
    ls_date, ls_symbols = read_data(orders_file)
    df_trade = create_trade_matrix(ls_date, ls_symbols)
    df_trade = update_trade_matrix(df_trade, orders_file)
    df_cash = create_cash_matrix(ls_date, begin_cash)
    df_cash = update_cash_matrix(df_cash, df_trade)
    df_holding = convert_trade_matrix_to_holding_matrix(df_trade)
    df_value = port_value(df_cash, df_holding)
    write_output(output_file, df_value)
    
    return df_value


if __name__ == '__main__':
    
    # List of symbols
    df_value = marketsim(100000, "bollinger_orders.csv", "bollinger_values.csv")
    analyzer.analyze(df_value)
    
