from scipy.io import loadmat
import numpy as np
import LR_GradientAscent as ga
import LR_PredictLabels as pl


mat = loadmat('HW4Data.mat')

xTrain = mat['XTrain']
yTrain = mat['yTrain'].reshape(-1,1)
xTest  = mat['XTest']
yTest = mat['yTest'].reshape(-1,1)

#Train logistic regression

wHat,objVals = ga.gradient_ascent(xTrain,yTrain);

#Test logistic regression
yHat,numErrors = pl.predict_labels(xTest,yTest,wHat);

#Print the number of misclassified examples
print('There were %d misclassified examples in the test set\n',numErrors);
