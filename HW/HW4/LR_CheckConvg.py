# The matlab code translated is originally authored by Tom Mitchell
# Translated to python 3 by Michael J. P. Morse Sept. 2019

# Check whether the objective value has converged
# by comparing the difference between consecutive
# objective values to the tolerance

# function hasConverged = LR_CheckConvg(oldObj,newObj,tol)
#
#   % compute difference between objectives
#   diff = abs(oldObj-newObj);
#
#   % compare difference to tolerance
#   if (diff < tol)
#     hasConverged = true;
#   else
#     hasConverged = false;
#   endif
#
# end

def check_convg(old_obj,new_obj,tol):
    import numpy as np
    #compute difference between old and new objectives
    diff = np.abs(old_obj-new_obj);
#    print(diff)
    if(diff < tol):
        has_converged = True;
    else:
        has_converged = False;

    return has_converged
