import numpy as np

def ME(y, t):
    return (y-t).mean(axis=None)

def MAE(y, t):
    return (abs(y - t)).mean(axis=None)

def MSE(y, t):
    return ((y-t)**2).mean(axis=None)

def SSE(y, t):
    return 0.5*np.sum((y-t)**2)

def MSLE(y, t):
    return np.log((y-t)**2).mean(axis=None)

def RMSE(y, t):
    return np.sqrt(((y - t) ** 2).mean(axis=None))

def RMSLE(y, t):
    return np.log(np.sqrt(((y - t) ** 2).mean(axis=None)))

def MPE(y, t):
    return (((y-t)/y)*100).mean(axis=None)

def MAPE(y, t):
    return ((abs((y-t)/y))*100).mean(axis=None)

def MASE(y, t):
    n = len(y)
    d = np.abs(np.diff(y)).sum() / (n - 1)
    errors = abs(y-t)
    return errors.mean(axis=None)/d
