import numpy as np
from sklearn import linear_model, metrics
import math
from tqdm import tqdm

# import X and y
X = np.load("./humpback-whale-identification/X.npy")
y = np.load("./humpback-whale-identification/y.npy")

# Reshape X to be flatter
X_flat = X.reshape(25361, 10000)

# Train the machine in batches
listOMachines = []
for i in tqdm(range(math.ceil(y.size/500))):
    listOMachines.append(linear_model.LogisticRegression(solver='saga', multi_class='multinomial'))
    if i != math.ceil(y.size/500)-1:
        listOMachines[i].fit(X_flat[i*500:(i+1)*500],y[i*500:(i+1)*500])
    else:
        listOMachines[i].fit(X_flat[i*500:25361],y[i*500:25361])

# Test on the training set
listOYhats = []
for i in tqdm(range(len(listOMachines))):
    listOYhats.append(listOMachines[i].predict(X_flat))
array = np.array(listOYhats)
yhat =  np.mean(array, axis=0).astype(int)

# Calculate loss and accuracy
loss = metrics.log_loss(y, yhat)
accuracy = np.mean(np.equal(np.argmax(y),np.argmax(yhat)))

print("Training loss: " + str(loss))
print("Training accuracy: " + str(accuracy))
