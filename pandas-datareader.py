# http://pandas-datareader.readthedocs.io/en/latest/remote_data.html#remote-data-google

import pandas_datareader.data as web

import datetime

start = datetime.datetime(2015, 2, 9)

end = datetime.datetime(2017, 5, 24)

f = web.DataReader('SPY', 'morningstar', start, end)

print f.head()