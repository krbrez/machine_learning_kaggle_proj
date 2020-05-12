import numpy as np
from sklearn import linear_model, metrics, preprocessing
import math
from tqdm import tqdm

# import X and y
X = np.load("./humpback-whale-identification/X.npy")
y = np.load("./humpback-whale-identification/y.npy")

# Reshape X to be flatter
X_flat = X.reshape(25361, 784)
scale = preprocessing.StandardScaler()
X_sta = scale.fit_transform(X_flat)

# Create and train the machine
listOModels = []
for i in tqdm(range(int(y.size/512))):
    listOModels.append(linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial'))
    listOModels[i].fit(X_sta[i*512:(i+1)*512],y[i*512:(i+1)*512])

# Test on the training set
listOYhats = []
for i in tqdm(range(len(listOModels))):
    listOYhats.append(listOModels[i].predict(X_sta))
array = np.array(listOYhats)
yhat =  np.mean(array, axis=0).astype(int) # Non-linear kernel

# Calculate loss and accuracy
loss = -(1 / X_sta.shape[1]) * np.sum(y * (np.log(yhat)))
accuracy = np.mean(np.equal(y,yhat))

print("Training loss: " + str(loss))
print("Training accuracy: " + str(accuracy))
