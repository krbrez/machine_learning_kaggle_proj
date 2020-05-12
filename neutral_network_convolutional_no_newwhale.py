# This code is the same as the convolutional neural network, but it is trained on only images that are labeled, no 'new_whale' images.

# Import some libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout
from keras.utils import to_categorical
import matplotlib .pyplot as plt
import pandas as pd
import cv2
import os
from tqdm import tqdm


# Get the data values from the csv
df = pd.read_csv("train_no_new_whale.csv")



# Convert the pandas datafram to a numpy array
nparry = df.to_numpy()
# There are 25361 images and corresponding labels(thus an array of 25361 by 2)
print(nparry.shape)



# Seperate the whale image IDs and the labels(y) into two seperate arrays
whale_image_IDs = nparry[:,0]
labels = nparry[:,1]



# Declare which folder has the training images
DATADIR = "./humpback-whale-identification"
category = "train"
path = os.path.join(DATADIR, category)



X_grey_100x100 = np.empty((15697, 100, 100))
print(X_grey_100x100.shape)



# We need to create a one-hot encoded thing
y, uniques = pd.factorize(labels)
print(labels[2])
print(y.shape) #[0, 1, 2, 3, 3, 3, 4 ...]
print(uniques.shape) #All unique labels



## This dataset will be black and white, and of size 100x100
index = 0
for img in tqdm(os.listdir(path)):
    if img in whale_image_IDs:
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (100, 100))
        row_vec = new_array[np.newaxis, :]
        X_grey_100x100[index] = row_vec
        #plt.imshow(new_array)
        #plt.show()
        index += 1
print(X_grey_100x100[0].shape)
print(X_grey_100x100.shape)
np.save('X_grey_100x100', X_grey_100x100)



# We need to create a one-hot encoded thing
y, uniques = pd.factorize(labels)
print(y) #[0, 1, 2, 3, 3, 3, 4 ...]
print(uniques) #All unique labels


# Apparently it is helpful to add a dimension
X_grey_100x100 = np.expand_dims(X_grey_100x100, axis=3)
print(X_grey_100x100.shape) #(25361, 100, 100, 1)
print(labels.shape) #(25361)


# We will normlize the image as this helps apparently
X_grey_100x100 = (X_grey_100x100 / 255) - 0.5



# We will have a conv layer, a maxpooling layer and a dense layer

num_filters = 20
filter_size = 5
pool_size = 5

model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(100, 100, 1)),
  MaxPooling2D(pool_size=pool_size),
  Dropout(0.5),
  Flatten(),
  Dense(5004, activation='softmax'),
])



# The model will use adam as the optimizer, categorical_cross entropy loss
from keras.optimizers import Adam
model.compile(
  optimizer=Adam(lr=0.005),
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)



hist = model.fit(
  X_grey_100x100, # training data
  to_categorical(y), # training targets
  epochs=5,
  batch_size=256,
  validation_split=0.3
)




# Save the model
model.save_weights('model15.h5')



#Visualize the models accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()



#Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


X_grey_100x100_test = np.empty((7960, 100, 100))

DATADIR_test = "./humpback-whale-identification"
category_test = "test"
path = os.path.join(DATADIR_test, category_test)

# This dataset will be black and white, and of size 100x100
index = 0
whale_ids = []
for img in tqdm(os.listdir(path)):
    whale_ids.append(img)
    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (100, 100))
    row_vec = new_array[np.newaxis, :]
    X_grey_100x100_test[index] = row_vec
    #plt.imshow(new_array)
    #plt.show()
    index += 1




#Flatten and normalize the data
#X_grey_100x100_test = X_grey_100x100_test.reshape((-1, 10000))
X_grey_100x100_test = np.expand_dims(X_grey_100x100_test, axis=3)
X_grey_100x100_test = (X_grey_100x100_test / 255) - 0.5
print(X_grey_100x100_test.shape)



# We will store our guesses in a pandas data frame for now
df = pd.DataFrame(columns=['Image', 'Id'])
# Declare which folder has the testing images


index = 0
for img in tqdm(X_grey_100x100_test):
    probabilities = model.predict(np.array([img]))
    #print(probabilities)
    target_index = np.where(probabilities[0] == np.amax(probabilities[0]))
    
    #Get the correct id for that index
    y_guess = uniques[target_index[0][0]]
    img_code = whale_ids[index] 
    df.loc[index] = img_code, y_guess
    index +=1



#Save the csv
df.to_csv('15th_try.csv', index=False)





