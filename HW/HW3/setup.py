from scipy.io import loadmat
import numpy as np
from NB_XGivenY import x_Given_y as xgy
from NB_YPrior import yPrior
from NB_Classify import classify
from ClassificationError import error

mat = loadmat('HW3Data.mat')

xTrain = mat['XTrain'].toarray()
yTrain = mat['yTrain'].flatten(1)
xTest  = mat['XTest'].toarray()
yTest = mat['yTest'].flatten(1)


p = yPrior(yTrain)
d = xgy(xTrain,yTrain)

yHat_train = classify(d,p,xTrain)
yHat_test = classify(d,p,xTest)

print(error(yHat_train,yTrain))
print(error(yHat_test,yTest))
