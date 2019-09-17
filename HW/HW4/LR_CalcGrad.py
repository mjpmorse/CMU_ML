# The matlab code translated is originally authored by Tom Mitchell
# Translated to python 3 by Michael J. P. Morse Sept. 2019

# Calculate the gradient of the logistic regression
# objective function with respect to each parameter

# function grad = LR_CalcGrad(XTrain,yTrain,wHat)
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
#   % calculate gradient
#   grad = sum(repmat(yTrain,1,p+1).*XTrain - XTrain.*repmat(eXw./(1+eXw),1,p+1))';
#
# end


def calc_grad(x_train,y_train,w_hat):
    import numpy as np
    import numpy.matlib as npml

    #get the dimensions of the independent array
    num_row,num_col = np.shape(x_train);

    #add a feature (column) of ones to x_train
    x_train = np.append(np.ones([num_row,1]),x_train,axis=1);

    #precompute x.w and exp(x,w)
    xw = np.dot(x_train,w_hat);
    exw = np.exp(xw);

    #calculate gradient
#    print( "t1 " ,np.shape(x_train),np.shape(w_hat))
#    print( "t2 " ,np.shape(np.tile(exw/(1+exw),(1,num_col+1))))
#    print(" t3", np.shape(npml.repmat(exw/(1+exw),1,num_col)),num_col)
    grad = sum(np.tile(y_train,(1,num_col+1))*x_train -
    np.tile(exw/(1+exw),(1,num_col+1))*x_train);
#    print(grad)


    return grad
