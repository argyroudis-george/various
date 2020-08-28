#Rescaled_Range_Analysis

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import math
import statistics as stat
import scipy.stats

#Functions
def average_function(alpha):
    avrg = sum(alpha)/len(alpha)
    return avrg

#data input, creation of DataFrame

#data_input = input('Please enter your dataset, including the .csv: ')
data_input = 'hsi86_16.csv'
dat = pd.read_csv(data_input, header=None, names=['dates','prices'])
df = pd.DataFrame(dat)
df['returns'] = df['prices'].pct_change()
df['returns'].fillna(0, inplace=True)

# loop of how many times we want to do the range
num_ranges = np.int(input('Please enter the number of range you wish: '))

#choose the ranges to be powers of 2 or powers of 3, the former divides the
#time series by 2,4,8,16,32 and the latter 3,9,27,81,243
two_or_three = np.int(input('Please enter the power you wish: '))

# list of ranges
size_of_range = list()
for i in range(num_ranges+1):
    size_of_range.append(math.floor(len(df['returns'])/np.power(two_or_three,i)))


# matrix with rows equal to the number of total ranges, and columns with pieces of the data on each column. For example,
# in the first row we have data only in the first column because the range of the time series is the whole length.
data_range = [[0 for j in range(np.int(len(df['returns'])/size_of_range[-1]))] for i in range(len(size_of_range))]
for i in range(len(size_of_range)):
    a,b = 0, size_of_range[i]
    for j in range(0, np.int(len(df['returns'])/size_of_range[i])):
        data_range[i][j] = (df['returns'][a:b]).to_list()
        a +=size_of_range[i]
        b +=size_of_range[i]

# average_matrix, the average of every range. Where there is no data in the matrix we use 0
average_matrix = [[0 for j in range(len(data_range[0]))] for i in range(len(data_range))]
for i in range(len(data_range)):
    for j in range(len(data_range[0])):
        if data_range[i][j]== 0:
            average_matrix[i][j] = 0
        else:
            average_matrix[i][j] = average_function(data_range[i][j]) #call the function average_function

# deviations matrix. data_range-mean, a new matrix where we subtract each element of the list with its respective mean
deviations_matrix = [[0 for j in range(len(average_matrix[0]))] for i in range(len(average_matrix))]
for i in range(len(average_matrix)):
    for j in range(len(average_matrix[0])):
        if average_matrix[i][j] ==0:
            deviations_matrix[i][j] = 0
        else:
            deviations_matrix[i][j] = []
            for k in range(len(data_range[i][j])): #loop mesa sti lista tis listas pou exei o data_range
                deviations_matrix[i][j].append(data_range[i][j][k] - average_matrix[i][j])

# sum of the deviations from each range in the deviations_matrix
sum_deviations = [[0 for j in range(len(deviations_matrix[0]))] for i in range(len(deviations_matrix))] # create a matrix
for i in range(len(deviations_matrix)):
    for j in range(len(deviations_matrix[0])):
        if isinstance(deviations_matrix[i][j],list):  #if what we read is a list, then sum the list
            sum_deviations[i][j] = sum(deviations_matrix[i][j]) #place the sum to the sum_deviations list
        else:
            sum_deviations[i][j] = 0

# find the difference between max and min from each range(list) in the matrix deviations_matrix
max_min_deviation = [[0 for j in range(len(deviations_matrix[0]))] for i in range(len(deviations_matrix))]
for i in range(len(deviations_matrix)):
    for j in range(len(deviations_matrix[0])):
        if isinstance(deviations_matrix[i][j],list):
            max_min_deviation[i][j] = max(deviations_matrix[i][j]) - min(deviations_matrix[i][j])
        else:
            max_min_deviation[i][j] = 0

# find the standard deviation from every range(list) in the matrix_deviations matrix
std_matrix = [[0 for j in range(len(deviations_matrix[0]))] for i in range(len(deviations_matrix))]
for i in range(len(deviations_matrix)):
    for j in range(len(deviations_matrix[0])):
        if isinstance(deviations_matrix[i][j],list):
            std_matrix[i][j] = stat.stdev(deviations_matrix[i][j])
        else:
            std_matrix[i][j] = 0

# Rescaled Range Analysis, matrix where divide range with std of each range
rs = [[0 for j in range(len(deviations_matrix[0]))] for i in range(len(deviations_matrix))]
for i in range(len(deviations_matrix)):
    for j in range(len(deviations_matrix[0])):
        if isinstance(deviations_matrix[i][j],list):
            rs[i][j] = max_min_deviation[i][j]/std_matrix[i][j]
        else:
            rs[i][j] = 0

# Average the rs, for every group of range, for instance in the first range we have no second measurement.
# For the second range, we find the average of two ranges, etc
count = 0
sum_avg_rs = 0
avg_rs = [0 for i in range(len(deviations_matrix))]
for i in range(len(rs)):
    for j in range(len(rs[0])):
        if rs[i][j] == 0:
            continue
        else:
            sum_avg_rs += rs[i][j]
            count += 1
    avg_rs[i] = sum_avg_rs/count

# calculate log10 values of the ranges chosen and the avg_rs
log10_ranges = [math.log10(i) for i in size_of_range]
log10_avg_rs = [math.log10(i) for i in avg_rs]
X = np.asarray(log10_ranges)
Y = np.asarray(log10_avg_rs)

#calculate the linear regression with the use of scipy.math.linregress
results = scipy.stats.linregress(X,Y)
print('The Rescaled Range Analysis is {}'.format(results.slope))
if results.slope <0.48:
    print('There is anti-persistent behavior')
elif results.slope>=0.48 and results.slope<=0.52:
    print('There is random behavior')
elif results.slope>0.52:
    print('There is persistent behavior')

#plot the data
plt.plot(X, Y, 'g--')
plt.plot(X, results.intercept + results.slope*X, 'r')
# plt.show()