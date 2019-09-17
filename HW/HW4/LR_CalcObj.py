# The matlab code translated is originally authored by Tom Mitchell
# Translated to python 3 by Michael J. P. Morse Sept. 2019

# % Calculate the logistic regression objective value
#
# function obj = LR_CalcObj(XTrain,yTrain,wHat)
#
#   % get dimensions
#   [n,p] = size(XTrain);
#
#   % add a feature of 1's to XTrain
#   XTrain = [ones(n,1) XTrain];
#
#   % precompute X*w and exp(X*w)
#   Xw = XTrain*wHat;
#   eXw = exp(Xw);
#
#   % calculate objective value
#   obj = sum(yTrain.*Xw - log(1 + eXw));
#
# end
def calc_obj(x_train,y_train,w_hat):
    import numpy as np
    # Get the dimensions of xTrain
    num_row,num_col = np.shape(x_train);
    # add a deature of 1's to xTrain
    x_train = np.append(np.ones([num_row,1]),x_train,axis=1);

    #precompute x.w and exp(x,w)
    xw = np.dot(x_train,w_hat);
    exw = np.exp(xw);

    #calculate the objective value

    obj = np.sum(xw*y_train - np.log(1 + exw));
    return obj
