#load the X_train day in to X_t and view data present in the array in human readable format
import numpy as np

X_t = np.load('X_train.npy')
print(X_t.shape)