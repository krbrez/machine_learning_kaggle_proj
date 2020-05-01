import numpy as np
import matplotlib .pyplot as plt
import os
import cv2
import pandas as pd
from tqdm import tqdm

# We read the csv Data in using pandas(this is the fastest way I could think to do this)
df = pd.read_csv("train.csv") 

# We convert the pandas datafram to a numpy array
nparry = df.to_numpy()
print(nparry.shape)

# We then seperate the whale image IDs and the labels into two seperate arrays
whale_image_IDs = nparry[:,0]
labels = nparry[:,1]

"""
# We need the labels to be in the correct form, so we do some witchcraft on them
yhat = np.zeros((labels.shape[0], np.unique(labels).shape[0]))
indexing_array = {}
index = 0
true_index = 0
for x in tqdm(labels):
	arr = np.zeros(np.unique(labels).shape[0])
	#third value is 1
	if x in indexing_array:
		#get the index of x
		target_index = indexing_array.get(x)
	else:
		indexing_array.update({x: index})
		target_index = index
		index += 1
	true_index +=1

	arr[target_index] = 1
	#print(arr)
	yhat[true_index - 1] = arr

print(yhat.shape)
print(labels.shape)

np.save('y', yhat)
"""
y_load = np.load("y.npy")
print(y_load.shape)

# For the time being we will convert the image to grey scale, and make them 200x200
"""
WIDTH = 100
HEIGHT = 100
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
"""
X = np.load("X.npy")
print(X.shape)


#If everything worked correctly, we should have two arrays, one of the images, one of the lables