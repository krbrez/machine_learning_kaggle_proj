import numpy as np
from sklearn import linear_model, metrics, preprocessing
import math
from tqdm import tqdm
import os
import json
import cv2
import pandas as pd

# Code integrated from Alex's part of the project
X_test = np.empty((7960, 28, 28))

DATADIR_test = "./humpback-whale-identification"
category_test = "test"
path = os.path.join(DATADIR_test, category_test)

index = 0
whale_ids = []
for img in tqdm(os.listdir(path)):
    whale_ids.append(img)
    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (28, 28))
    row_vec = new_array[np.newaxis, :]
    X_test[index] = row_vec

    index += 1

# We will store our guesses in a pandas data frame for now
df = pd.DataFrame(columns=['Image', 'Id'])

# Start my code
# import X and y
X = np.load("./humpback-whale-identification/X.npy")
y = np.load("./humpback-whale-identification/y.npy")

# Reshape X to be flatter
X_flat = X.reshape(25361, 784)
X_sta = preprocessing.scale(X_flat)

# Create and train the machine
listOModels = []
for i in tqdm(range(int(y.size/2048))):
    listOModels.append(linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial'))
    listOModels[i].fit(X_sta[i*2048:(i+1)*2048],y[i*2048:(i+1)*2048])

# Test on the training set
listOYhats = []
for i in tqdm(range(len(listOModels))):
    listOYhats.append(listOModels[i].predict(X_sta))
array = np.array(listOYhats)
yhat =  np.mean(array, axis=0).astype(int)

# Calculate loss and accuracy
loss = -(1 / X_sta.shape[1]) * np.sum(y * (np.log(yhat)))
accuracy = np.mean(np.equal(y,yhat))

print("Training loss: " + str(loss))
print("Training accuracy: " + str(accuracy))

# Load in the legend
with open("./humpback-whale-identification/legend.json", "r") as js:
	legend = json.load(js)

# Convert to 2 arrays because this would be a reverse dictionary lookup which it isn't set up for
ids = list(legend.keys())
vals = list(legend.values())

# Reshape X_test to be flatter
X_t_flat = X_test.reshape(7960, 784)
X_t_sta = preprocessing.scale(X_t_flat)

# Test on the test set for uploading to Kaggle
listOYhatsa = []
for i in tqdm(range(len(listOModels))):
    listOYhatsa.append(listOModels[i].predict(X_t_sta))
array2 = np.array(listOYhatsa)
yhat2 =  np.mean(array2, axis=0).astype(int)
index = 0
for guess in tqdm(yhat2):
    ind = vals.index(guess)
    id = ids[ind]
    df.loc[index] = [whale_ids[index], id]
    index += 1

df.to_csv('submission.csv', index=False)
