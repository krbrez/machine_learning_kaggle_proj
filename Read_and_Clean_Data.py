import numpy as np
import matplotlib .pyplot as plt
import os
import cv2
import pandas as pd
from tqdm import tqdm
import json

# We read the csv Data in using pandas(this is the fastest way I could think to do this)
df = pd.read_csv("./humpback-whale-identification/train.csv")

# We convert the pandas datafram to a numpy array
nparry = df.to_numpy()
print(nparry.shape)

# We then seperate the whale image IDs and the labels into two seperate arrays
whale_image_IDs = nparry[:,0]
labels = nparry[:,1]
print(labels)
print(labels.shape)

# Convert labels to numbers and create a legend to save that matches up the number to the label
legend = {}
yhat = np.zeros(labels.shape[0])
index = 0
new_assignment = 0
for x in tqdm(labels):
	if x in legend.keys():
		yhat[index] = legend[x]
	else:
		legend[x] = new_assignment
		yhat[index] = new_assignment
		new_assignment += 1
	index += 1


print(yhat)
print(yhat.shape)
print(labels.shape)

np.save('y', yhat)
with open("legend.json", "w") as js:
	json.dump(legend, js)

y_load = np.load("y.npy")
print(y_load.shape)

# For the time being we will convert the image to grey scale, change width and height as needed to test different image sizes

WIDTH = 28
HEIGHT = 28
DATADIR = "./humpback-whale-identification"
category = "train"
X = np.zeros(shape=(25361, WIDTH, HEIGHT))

path = os.path.join(DATADIR, category)
index = 0
for img in tqdm(os.listdir(path)):
	try:
		img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
		new_array = cv2.resize(img_array, (WIDTH, HEIGHT))
		X[index] = new_array
		#plt.imshow(new_array, cmap="gray")
		#plt.show()
		index += 1
	except IndexError:
		print("We couldn't find that image")
#print(X)
np.save('X', X)

X = np.load("X.npy")
print(X.shape)


#If everything worked correctly, we should have two arrays, one of the images, one of the lables
