import numpy as np
from sklearn import linear_model, datasets

# import X and y
X = np.load("./humpback-whale-identification/X.npy")
y = np.load("./humpback-whale-identification/y.npy")

# Reshape X to be flatter
X_flat = X.reshape(25361, 10000)

# Make the machine
softmax = linear_model.LogisticRegression(solver='saga', multi_class='multinomial')

# Train the machine
softmax.fit(X_flat, y)

# Test on the training set
yhat = softmax.predict(X_flat)

# Calculate loss and accuracy
loss = -(1 / X_flat.shape[0]) * np.sum(y * (np.log(yhat)))
accuracy = np.mean(np.equal(np.argmax(y, axis=0),np.argmax(yhat, axis=0)))

print("Training loss: " + str(loss))
print("Training accuracy: " + str(accuracy))
