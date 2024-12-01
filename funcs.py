from libs import *

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps].values)
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)