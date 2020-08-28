import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tabulate import tabulate
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# #download the dataset
# dataset = pd.read_csv('hsi86_16.csv',header=None,names=['dates','prices'])
# df = pd.DataFrame(dataset)
# df['returns'] = df['prices'].pct_change()
# df['returns'].dropna(inplace=True)


#
# nifty = pd.read_csv('nifty.csv', header=0 ,  index_col=0, parse_dates=True) #Get nifty prices
# nifty_ret = nifty['Close'].resample('W').last().pct_change().dropna() #Get weekly returns

# check series stationarity
# print(adfuller(nifty_ret.dropna()))

# Markov Switching Model
# model = sm.tsa.MarkovRegression(nifty_ret.dropna(), k_regimes=3,trend='nc', switching_variance=True )
# results = model.fit()
# results.summary()
ret = pd.read_csv('korean_data.csv', header=None)
# plt.plot(ret, 'g')
# plt.show()

print(adfuller(ret))
model = sm.tsa.MarkovRegression(ret,k_regimes=3,trend='nc',switching_variance=True)
results =model.fit()
print(results.summary())
plt.plot(results.smoothed_marginal_probabilities[2], 'g')
plt.show()