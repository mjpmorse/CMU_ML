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

def prior(theta):
    return (theta)**2 * (1-theta)**2 *(1./0.0333)


array_length = int((theta_end - theta_0)/step)
theta = theta_0

likelyhood_arrayMLE = np.zeros([array_length,2])
likelyhood_arrayMAP = np.zeros([array_length,2])
i = 0
s = 6
t = 10
while(theta<theta_end):
    likelyhood_arrayMLE[i,0] = theta
    likelyhood_arrayMAP[i,0] = theta
    likelyhood_arrayMLE[i,1] = likelyhood(theta,s,t)
    likelyhood_arrayMAP[i,1] = likelyhood(theta,s,t)*prior(theta)
    i +=1
    theta += step

plt.plot(likelyhood_arrayMLE[:,0],likelyhood_arrayMLE[:,1],label="MLE")
plt.plot(likelyhood_arrayMAP[:,0],likelyhood_arrayMAP[:,1],label="MAP")
plt.legend()
plt.xlabel('Theta')
plt.ylabel('L(Theta)')
plt.show()
