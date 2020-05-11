import numpy as np
from sklearn import linear_model, metrics
import math

# import X and y
X = np.load("./humpback-whale-identification/X.npy")
y = np.load("./humpback-whale-identification/y.npy")

# Reshape X to be flatter
X_flat = X.reshape(25361, 784)

# Create and train the machine
softmax = linear_model.LogisticRegression(solver='saga', multi_class='multinomial')
softmax.fit(X_flat, y)

# Test on the training set
yhat = softmax.predict(X_flat)

# Calculate loss and accuracy
loss = metrics.log_loss(y, yhat)
accuracy = np.mean(np.equal(y,yhat))

print("Training loss: " + str(loss))
print("Training accuracy: " + str(accuracy))
