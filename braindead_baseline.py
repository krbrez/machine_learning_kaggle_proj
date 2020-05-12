import numpy as np
import json
import pandas as pd
import cv2
from tqdm import tqdm
import os

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

# Commented code for testing on the training set
# Start my code
# # import X and y
# X = np.load("./humpback-whale-identification/X.npy")
# y = np.load("./humpback-whale-identification/y.npy")
#
# # Reshape X to be flatter
# X_flat = X.reshape(25361, 784)
#
# Open the legend
with open("./humpback-whale-identification/legend.json","r") as js:
    legend = json.load(js)
#
# # Use numpy to select a random class for each element of X
# classes = np.array(list(legend.values()))
# yhat = np.random.choice(classes, size=X_flat.shape[0])
#
# # Find loss and accuracy
# loss = -(1 / X.shape[1]) * np.sum(y * (np.log(yhat)))
# accuracy = np.mean(np.equal(y, yhat))
# print("Loss: " + str(loss))
# print("Accuracy: " + str(accuracy))

# # Reshape X test to be flatter
X_flat = X_test.reshape(7960, 784)
# # Use numpy to select a random class for each element of X
classes = np.array(list(legend.keys()))
yhat = np.random.choice(classes, size=X_flat.shape[0])

index = 0
for guess in tqdm(yhat):
    id = guess
    df.loc[index] = [whale_ids[index], id]
    index += 1

df.to_csv('submission.csv', index=False)
