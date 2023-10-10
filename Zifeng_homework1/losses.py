import numpy as np

class ValueLengthError(Exception):
    pass

# loss function and its derivative
def mse(y_true, y_pred):
    #TODO: return mse
    if len(y_true) != len(y_pred):
        raise ValueLengthError("Sorry, y_true and y_pred should have same length!")
    else:
        return np.mean((y_true - y_pred)**2)

def mse_prime(y_true, y_pred):
    #TODO: return mse derivative
    return -2*(y_true - y_pred) / y_true.shape[0]