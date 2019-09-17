# The matlab code translated is originally authored by Tom Mitchell
# Translated to python 3 by Michael J. P. Morse Sept. 2019

# Calculate the new value of wHat using the gradient
# ascent update rule

# function wHat = LR_UpdateParams(wHat,grad,eta)
#
#   % update value of w
#   wHat = wHat + eta.*grad;
#
# end

def update_params(w_hat,grad,eta):
     import numpy as np
     what = np.add(w_hat, eta*grad.reshape(-1,1));
#     print("w_hat",np.shape(w_hat),np.shape(grad))
#     print("what ", np.shape(what))
     return what
