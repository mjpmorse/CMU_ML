def yPrior(yTrain):
    p = sum(yTrain == 1)/len(yTrain);
    return p
