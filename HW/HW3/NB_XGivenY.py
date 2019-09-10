# The matlab code translated is originally authored by Tom Mitchell
# Translated to python 3 by Michael J. P. Morse Sept. 2019

# function [D] = NB_XGivenY(XTrain, yTrain)
#   EconoRows = yTrain == 1;
#   OnionRows = yTrain == 2;
#
#   D = [(sum(XTrain(EconoRows,:), 1) .+ 1) / (sum(EconoRows) + 1) ;
#        (sum(XTrain(OnionRows,:), 1) .+ 1) / (sum(OnionRows) + 1)];
# end


def x_Given_y(xTrain,yTrain):
    import numpy as np
    # #num_row is the number of samples
    # num_col is the number of vocab,V
    num_row,num_col = np.shape(xTrain)
    # need to return a 2xV array
    x_given_y = np.zeros([2,num_col])
    i_col = 0
    #we will deal with each word in vocab list one at a time
    while (i_col < num_col):
        i_row = 0
        # we are applying a prior on each word of 1 to insure no zeros in
        # the product. (1. to ensure double devision not int)
        real_vocab  = 1.;
        fake_vocab = 1.;
        num_real = 1.;
        num_fake = 1.;
        # for each word, i_col in V, we iterate over the rows each will be
        # either a fake news(yTrain==2) or real news source (yTrain==1).
        while (i_row < num_row):
            #check if the news is real
            if(yTrain[i_row] == 1):
                #if real news we will add it to the occurance in real
                num_real += 1
                real_vocab += xTrain[i_row,i_col]

            #else the news is fake
            else:
                num_fake +=1
                fake_vocab += xTrain[i_row,i_col]
            #iterate to the next row
            i_row += 1
        #return the elements of the 2xV matrix for i_col in V.
        x_given_y[:,i_col] = real_vocab/num_real  , fake_vocab/num_fake
        #increment the column
        i_col +=1
        if(i_col%1000 == 0):
            print(i_col)
    return x_given_y
