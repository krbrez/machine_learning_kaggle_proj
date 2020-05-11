import numpy as np
import pickle

# import X and y
X = np.load("./humpback-whale-identification/X.npy")
y = np.load("./humpback-whale-identification/y.npy")

# Reshape X to be flatter
X_flat = X.reshape(25361, 784)

# Unpickle the legend
pick = open("./humpback-whale-identification/legend.p","rb")
legend = pickle.load(pick)

# Use numpy to select a random class for each element of X
classes = np.ndarray(list(pick.values()))
yhat = np.random.choice(classes, size=X_flat.shape[0])

loss = metrics.log_loss(y, yhat)
accuracy = np.mean(np.equal(y, yhat))
