# # preprocess accelerometer data for gesture recognition
# # load hi.csv and sup.csv with pandas
# # normalize using sklearn MinMaxScaler (range -1 to 1)
# # label hi as 0, sup as 1
# # combine arrays and split with sklearn train_test_split
# # save X_train.npy, y_train.npy, X_test.npy, y_test.npy

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# # load hi.csv and sup.csv
# hi_data = pd.read_csv('hi.csv').values
# sup_data = pd.read_csv('sup.csv').values

# # normalize using MinMaxScaler (range -1 to 1)
# scaler = MinMaxScaler(feature_range=(-1, 1))
# hi_data = scaler.fit_transform(hi_data)
# sup_data = scaler.fit_transform(sup_data)

# # label hi as 0, sup as 1
# hi_labels = np.zeros((hi_data.shape[0],))
# sup_labels = np.ones((sup_data.shape[0],))

# # combine arrays
# X = np.vstack((hi_data, sup_data))
# y = np.hstack((hi_labels, sup_labels))

# # split with train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # save arrays
# np.save('X_train.npy', X_train)
# np.save('y_train.npy', y_train)
# np.save('X_test.npy', X_test)
# np.save('y_test.npy', y_test)   

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# # Load CSVs (no header assumed)
# hi = pd.read_csv("data/hi.csv").values.astype(np.float32)
# sup = pd.read_csv("data/sup.csv").values.astype(np.float32)

# # Labels: hi=0, sup=1
# y_hi = np.zeros(hi.shape[0], dtype=np.int64)
# y_sup = np.ones(sup.shape[0], dtype=np.int64)

# # Combine
# X = np.vstack((hi, sup))
# y = np.concatenate((y_hi, y_sup))

# # Normalize features to [-1, 1]
# scaler = MinMaxScaler(feature_range=(-1, 1))
# X_scaled = scaler.fit_transform(X)

# # Split (stratify to keep class proportions)
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42, stratify=y
# )

# # Save .npy files
# np.save("X_train.npy", X_train)
# np.save("y_train.npy", y_train)
# np.save("X_test.npy", X_test)
# np.save("y_test.npy", y_test)

# print("Saved arrays. Shapes:",
#       X_train.shape, y_train.shape, X_test.shape, y_test.shape)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 50  # how many consecutive rows make up one gesture sample

def make_windows(data, window_size):
    """Cut a continuous (rows, channels) stream into non-overlapping windows."""
    n_windows = len(data) // window_size
    data = data[:n_windows * window_size]  # trim leftover rows that don't fill a full window
    return data.reshape(n_windows, window_size, data.shape[1])

# Load CSVs
hi = pd.read_csv("data/hi.csv").values.astype(np.float32)
sup = pd.read_csv("data/sup.csv").values.astype(np.float32)

# Normalize features to [-1, 1] (same as before)
scaler = MinMaxScaler(feature_range=(-1, 1))
hi = scaler.fit_transform(hi)
sup = scaler.fit_transform(sup)

# NEW: cut each continuous stream into windows -> shape becomes (num_windows, WINDOW_SIZE, 6)
hi_windows = make_windows(hi, WINDOW_SIZE)
sup_windows = make_windows(sup, WINDOW_SIZE)

# Labels: hi=0, sup=1 (now one label per WINDOW, not per row)
y_hi = np.zeros(hi_windows.shape[0], dtype=np.int64)
y_sup = np.ones(sup_windows.shape[0], dtype=np.int64)

# Combine
X = np.vstack((hi_windows, sup_windows))
y = np.concatenate((y_hi, y_sup))

# Split (same as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save .npy files
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Saved arrays. Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)