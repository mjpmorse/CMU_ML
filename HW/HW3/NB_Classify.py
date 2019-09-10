# The matlab code translated is originally authored by Tom Mitchell
# Translated to python 3 by Michael J. P. Morse Sept. 2019

#   m = size(XTest, 1);
#   yHat = zeros(m, 1);
#
#   for i = 1:m
#     econo_probs = D(1,:) .* XTest(i,:) + (1 - D(1,:)) .* (1 - XTest(i,:));
#     onion_probs = D(2,:) .* XTest(i,:) + (1 - D(2,:)) .* (1 - XTest(i,:));
#
#     econo_score = logProd([log(econo_probs), log(p)]);
#     onion_score = logProd([log(onion_probs), log(1-p)]);
#
#     if econo_score > onion_score
#       yHat(i) = 1;
#     else
#       yHat(i) = 2;
#     end
#   end
# end

def classify(D,p,xTest):
    import numpy as np
    import logProd as lp
    #need to get the number of training events,num_row
    num_row,num_col = np.shape(xTest)
    #construct an output array which is num_data by 1
    yHat = np.zeros([num_row])
    i = 0
    while(i<num_row):
        # we construct the probability of each word being in real vs fake
        # news.

        econo_probs = D[0,:] * xTest[i,:] + (1.- D[0,:])*(1.-xTest[i,:])
        onion_probs = D[1,:] * xTest[i,:] + (1.- D[1,:])*(1.-xTest[i,:])

        econo_score = lp.log_product(np.log(np.append(econo_probs,p)))
        onion_score = lp.log_product(np.log(np.append(onion_probs,1-p)))
        if (econo_score > onion_score):
            yHat[i] = 1
        else:
            yHat[i] = 2
        i += 1
    return yHat
