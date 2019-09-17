# The matlab code translated is originally authored by Tom Mitchell
# Translated to python 3 by Michael J. P. Morse Sept. 2019

# Run the gradient ascent algorithm for logistic regression

def gradient_ascent(x_train,y_train):
    import numpy as np
    from LR_CalcObj import calc_obj
    from LR_GradientAscent import gradient_ascent
    from LR_UpdateParams import update_params
    from LR_CheckConvg import check_convg
    from LR_CalcGrad import calc_grad
    #set the step size
    eta = 0.01;

    #set the tolerance
    tol = 0.001;

    #get dimensions of input
    num_row,num_col = np.shape(x_train);

    #initalize w_hat
    w_hat = np.zeros([num_col+1,1]);

    #initialize objVals
    obj_vals = np.array([calc_obj(x_train,y_train,w_hat)]);

    #initalize convergence flag and check
    has_converged = False;

    #Run gradient ascent until convergence

    while( not has_converged):
        # Calculate gradient
        grad = calc_grad(x_train,y_train,w_hat);

        # Update parameter estimate
        w_hat = update_params(w_hat,grad,eta);

        # Calculate new objective
        new_obj = calc_obj(x_train,y_train,w_hat);

        # Check convergence
        has_converged = check_convg(obj_vals[-1],new_obj,tol);

        #store new objectives

        obj_vals = np.append(obj_vals,new_obj);
#        print(obj_vals)

    return w_hat , obj_vals
