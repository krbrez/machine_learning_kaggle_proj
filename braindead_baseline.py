import numpy as np
import json

# import X and y
X = np.load("./humpback-whale-identification/X.npy")
y = np.load("./humpback-whale-identification/y.npy")

# Reshape X to be flatter
X_flat = X.reshape(25361, 784)

# Unpickle the legend
with open("./humpback-whale-identification/legend.json","r") as js:
    legend = json.load(js)

# Use numpy to select a random class for each element of X
classes = np.array(list(legend.values()))
yhat = np.random.choice(classes, size=X_flat.shape[0])

# Find loss and accuracy
loss = -(1 / X.shape[1]) * np.sum(y * (np.log(yhat)))
accuracy = np.mean(np.equal(y, yhat))
print("Loss: " + str(loss))
print("Accuracy: " + str(accuracy))
