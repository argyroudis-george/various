import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from scipy.stats import norm

#Read the .csv file
data = pd.read_csv('hsi86_16.csv', header=None, names=['dates','prices'])
df = pd.DataFrame(data)
df['returns'] = df['prices'].pct_change(periods=1)
df.fillna(value=0, inplace=True)
sorted_df = df.sort_values(by='returns')
plt.hist(sorted_df['returns'],bins=40)
plt.show()
# quantile of the variable sorted_df['returns']

VaR90 = sorted_df['returns'].quantile(q=0.1)
VaR95 = sorted_df['returns'].quantile(q=0.05)
VaR99 = sorted_df['returns'].quantile(q=0.01)

#Call the tabulate
print(tabulate([['VaR90',VaR90],['VaR95',VaR95],['VaR99', VaR99]], headers=['Method', 'Measurement']))
