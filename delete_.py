import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize.optimize as opt
import scipy.optimize as opt1

# x = np.linspace(-10,10,51)
#
# f = lambda x: x**2 + 10*np.sin(x)
#
# y = f(x)
# plt.plot(x,y)
#
# res = opt.fmin_bfgs(f,x0=7.5,maxiter=2000, full_output=True)
# x_initial_guess = 0
# y_initial_guess = f(0)
# plt.scatter(x_initial_guess,y_initial_guess, color='r',marker='*')
#
# x_optimized = res[0]
# y_optimized = res[1]
# plt.scatter(x_optimized, y_optimized, color = 'r', marker='*')
#
# plt.show()

# binary = format(11, 'b')
# print(binary)
# print(binary[-1])


# class consoles:
#
#     chip_inside = 'affirmative'
#
#     def __init__(self, decade, color, shape, company):
#         self.decade = decade
#         self.color = color
#         self.shape = shape
#         self.company = company
#         self.__under_my_possession = 3
#
# snes = consoles('90s','grey','round','nintendo')
#
# print(snes.decade)
# print(snes._consoles__under_my_possession)
#
# mega_drive = consoles('90s','black','round','sega')
# mega_drive.chip_inside = 'blast_processing'
# print(snes.chip_inside)
# print(mega_drive.chip_inside)
#
# with open('little_file','w') as pp:
#     pp.write('the msm model and the multifractality')
#
# with open('little_file','r') as go:
#     print(go.read())
#
# binary = format(10,'b')
# print(binary)
#


# def cusum(x,mean=0,K=0):
#     """Tabular CUSUM per Montgomery,D. 1996 "Introduction to Statistical Process Control" p318
#     x    : series to analyze
#     mean : expected process mean
#     K    : reference value, allowance, slack value-- suggest K=1/2 of the shift to be detected.
#
#     Returns:
#     x  Original series
#     Cp positive CUSUM
#     Cm negative CUSUM
#     """
#     Cp=(x*0).copy()
#     Cm=Cp.copy()
#     for ii in np.arange(len(x)):
#         if ii == 0:
#             Cp[ii]=x[ii]
#             Cm[ii]=x[ii]
#         else:
#             Cp[ii]=np.max([0,x[ii]-(mean+K)+Cp[ii-1]])
#             Cm[ii]=np.max([0,(mean-K)-x[ii]+Cm[ii-1]])
#     return({'x':x, 'Cp': Cp, 'Cm': Cm})

# res = cusum(np.random.randn(10) + np.repeat([0, 0.8], 5),K=0.5)
# print(res)

# pd.DataFrame(cusum(np.random.randn(200)+np.repeat([0,0.5],100),K=0.25)).plot()
# plt.show()


data = pd.read_csv('hsi86_16.csv', names=['dates', 'prices'])
df = pd.DataFrame(data)

# create the sorted retuns
df['sorted_returns'] = df.prices.pct_change()
df = df[1:]
# df['sorted_returns'] = sorted_returns
print(df.head())

# print the length of the time series
ts_length = len(df.sorted_returns)

# sort the returns column only
df['sorted_returns'] = np.sort(df['sorted_returns'])

# drop the column of prices
df.drop('prices',axis=1, inplace=True)

# find the VaR90, VaR95 and VaR99 respectively
VaR90_position = np.int(np.round(0.1 * ts_length))
VaR95_position = np.int(np.round(0.05 * ts_length))
VaR99_position = np.int(np.round(0.01 * ts_length))

VaR90 = df.sorted_returns[VaR90_position]
VaR95 = df.sorted_returns[VaR95_position]
VaR99 = df.sorted_returns[VaR99_position]

print(VaR90, VaR95, VaR99)














