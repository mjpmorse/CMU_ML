# function [error] = ClassificationError(yHat, yTruth)
#   error = sum(yHat != yTruth) / length(yTruth);
# end

def error(yHat, yTruth):
    err = sum(yHat != yTruth)/len(yTruth);
    return err
