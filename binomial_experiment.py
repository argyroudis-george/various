# binomial_experiment: We apply 1000 times a 10 times coin toss test, and we see
# each time if its heads or tails. If the random is >0.5, we consider it as heads
# then we count how many heads we have on each trial, and we create a histogram

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# trials
trials= 1000
# number of independent experiments n
n = 10
# probability of each toss
p = 0.5

# Function that runs our coin toss trials
# heads is a list of the number of successes from each trial of n experiments

def binofunc(trials, n, p):
    heads = []
    for i in range(trials):  #thousand times loop
        tosses = list()
        for j in range(n):   #experiment of ten tosses
            tosses.append(np.random.random())
        counting = 0
        for m in range(len(tosses)):
            if tosses[m] > 0.5:
                counting +=1
        heads.append(counting)
    return heads

heads = binofunc(trials, n, p)
print(heads)
plt.hist(heads)
plt.show()



