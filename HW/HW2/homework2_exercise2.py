import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp


#Create a plot of \theta vs L(\theta)

step = 0.01
theta_0 = 0
theta_end = 1


def likelyhood(theta,success,total):
    binomial = sp.binom(total,success)
    failure = total - success
    return (theta)**success * (1-theta)**failure


array_length = int((theta_end - theta_0)/step)
theta = theta_0

likelyhood_array = np.zeros([array_length,2])
i = 0
s = 5
t = 10
while(theta<theta_end):
    likelyhood_array[i,0] = theta
    likelyhood_array[i,1] = likelyhood(theta,s,t)
    i +=1
    theta += step

plt.plot(likelyhood_array[:,0],likelyhood_array[:,1])
plt.xlabel('Theta')
plt.ylabel('L(Theta)')
plt.show()
