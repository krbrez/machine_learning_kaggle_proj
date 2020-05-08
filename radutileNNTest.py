import numpy as np
from sklearn.neural_network import MLPClassifier


# import X and y
X = np.load("./humpback-whale-identification/X.npy")
y = np.load("./humpback-whale-identification/y.npy")

# Reshape X to be flatter
X_flat = X.reshape(25361, 10000)  # note, adjust later code based on which dimension is the examples.

indices = np.arange(len(y))

sets = np.split(indices, 3)

Xtrain = X_flat[sets[0], :]
Xvalid = X_flat[sets[1], :]
Xtest = X_flat[sets[2], :]

ytrain = y[sets[0]]
yvalid = y[sets[1]]
ytest = y[sets[2]]

classifier = MLPClassifier()  # hyperparameters can be adjusted here.
classifier.fit(Xtrain, ytrain)

yhat = classifier.predict(Xvalid)

# check accuracy of yhat here:
