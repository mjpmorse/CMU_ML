# The matlab code translated is originally authored by Tom Mitchell
# Translated to python 3 by Michael J. P. Morse Sept. 2019

# Predict the labels for a test set using logistic regression



def predict_labels(x_test,y_test,w_hat):
    import numpy as np

    #get dimensions
    num_row,num_col = np.shape(x_test)

    #add a feature of 1's to x_test

    x_test = np.append(np.ones([num_row,1]),x_test,axis=1);

    #precompute x.w and exp(x,w)
    xw = np.dot(x_test,w_hat);
    exw = np.exp(xw);

    # calculate p(Y=0)
    py0 = 1./(1+ exw);

    # calculate p(Y=1)
    py1 = exw/(1+ exw);

    #choose best label
    prob = np.append(py0,py1,axis=1);
    y_hat = np.argmax(prob,axis=1).reshape(-1,1);

    num_error = sum(y_hat != y_test);
    return y_hat, num_error
