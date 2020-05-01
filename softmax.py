import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from skimage.transform import rotate
from skimage.transform import resize
import random


def randomNoise(image):
    transformation = np.copy(trainingImages[0].reshape(28,28))
    randomValues = np.random.rand(28, 28)
    randomValues = randomValues / 3
    ret = transformation + randomValues
    return ret


def rotateImage(image):
    transformation = np.copy(trainingImages[0].reshape(28,28))
    return rotate(transformation, 9)

def scaleImage(image):
    transformation = np.copy(trainingImages[0].reshape(28,28))
    return resize(transformation, (28,28))

def translateImage(image):
    transformation = np.copy(trainingImages[0].reshape(28,28))
    zeros = np.zeros((transformation.shape[0], 1))
    x = np.concatenate((zeros,transformation), axis = 1)
    ret = np.delete(x, 28, axis=1)
    return ret


# Calculate the yhat value
def getyhat(sub_ImageArray, oldW):

    Z = np.dot(sub_ImageArray, oldW)
    yhat = (np.exp(Z).T / np.sum(np.exp(Z), axis=1))
    return yhat.T

# We are calulating the percent correct based on the yhat and the y, ideally, this number will go up over time
def calculatePC(oldW, trainingImages_shuffled, trainingLabels_shuffled):
    Yhat = getyhat(trainingImages_shuffled, oldW)
    Y = trainingLabels_shuffled

    #Initilize the varaibles we need to keep track of things
    correct = 0
    for x in range(Y.shape[0]):
        filtered = np.zeros(5005)

        # We use argemax to get the index of the highest value in the array, and
        # we then replace it with a 1, so we can properly compare arrays
        filtered[np.argmax(Yhat[x])] = 1

        if (np.array_equal(Y[x], filtered)):
            correct += 1

    return correct / Y.shape[0]


# We need to calculate the loss to make sure we are getting better each epoch
def calculateFCE(oldW, trainingImages_shuffled, trainingLabels_shuffled):
    Yhat = getyhat(trainingImages_shuffled, oldW)
    Y = trainingLabels_shuffled

    #print("This is Yhat shape: ", Yhat.shape)
    #print(Yhat[0])
    #print("This is Y shape: ", Y.shape)

    runningCount = 0
    for x in range(Y.shape[0]):
        runningCount += np.dot(Y[x], np.log(Yhat[x]))
        
    return - runningCount

# We need to calculate the gradient using formula 1/n * X * (Yhat - Y)
def getGradient(oldW, sub_ImageArray, subLabelsArray):
    Yhat = getyhat(sub_ImageArray, oldW)
    Y = subLabelsArray


    var1 = Yhat - Y
    var2 = sub_ImageArray / sub_ImageArray.shape[0]

    # If my numbers come out strange it is probably due to this part, I am 
    # translating the var2 because that is the only way the arethetic would work 
    return np.dot(var2.T, var1)




# Get the optimized W value by doing gradient decent on them
def stochasticGradientDescent(W, trainingImages, trainingLabels, batchSize, epsilon):
    # Randomize the training data and training labels, but make sure they have the
    # same random "seed". We can emulate this seed by shuffling an array of index
    # values and then applying those index values to both arrays
    s = np.arange(trainingImages.shape[0])
    np.random.shuffle(s)

    trainingImages_shuffled = np.copy(trainingImages[s])
    trainingLabels_shuffled = np.copy(trainingLabels[s])

    # Let's say the tolerance is 0.01
    tolerance = 0.01
    # For each epoch (e=1 ... E or until tolerance is reached)
    epochCount = 0
    oldFCE = 0
    newFCE = 100
    oldPC = 0
    newPC = 100
    oldW = W
    #while abs(oldFCE - newFCE) > tolerance:
    for j in range(10):
        epochCount += 1
        oldFCE = newFCE
        oldPC = newPC

        # For each round (r=1 ... r=ceil(n/batchSize) which is 25361 / 3623 or 7)
        for x in range(math.ceil(trainingImages.shape[0] / batchSize)):
            # Get the 100 images that we need, and the coresponding y values for them
            lower = x * batchSize
            upper = x * batchSize + batchSize

            # At this point we have a minibatch of 100 images, and their 100 corresponding
            # labels. We can use this to optimize the weights using gradient decent
            sub_ImageArray = np.copy(trainingImages_shuffled[lower:upper])
            subLabelsArray = np.copy(trainingLabels_shuffled[lower:upper])

            newW = oldW - epsilon * getGradient(oldW, sub_ImageArray, subLabelsArray)
            oldW = np.copy(newW)

        # Calculate the Cross Entropy loss
        newFCE = calculateFCE(oldW, trainingImages_shuffled, trainingLabels_shuffled)
        newPC = calculatePC(oldW, trainingImages_shuffled, trainingLabels_shuffled)
        print("NewFCE vs OldFCE: ", newFCE, oldFCE)
        print("NewPC vs OldPC: ", newPC, oldPC)

    print("Completed in -> " + str(epochCount) + " epochs")

    return oldW

# Add a 1 to the end of each image so now it is 785 numbers long
def addBias(images) :
    return np.hstack((images, np.atleast_2d(np.ones(images.shape[0])).T))

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):
    # Step 1: Initilize the weights to random values
    W = 0.01 * np.random.randn(10001, 5005)

    # Step 2: Optimize W using stochastic gradient descent
    optimized_W = stochasticGradientDescent(W, trainingImages, trainingLabels, batchSize, epsilon)
    return optimized_W

if __name__ == "__main__":
    # Load data
    X = np.load("X.npy")
    y = np.load("y.npy")

    # There are 25,362 images in the training set
    # Labels are organized as [1, 0, 0, 0 ... 0, 0, 0, 0, 0] with a 1 denoting if it is that item

    # Reshape X to be flatter
    X_flat = X.reshape(25361, 10000)

    # Append a constant 1 term to each example to correspond to the bias terms
    X_b = addBias(X_flat)

    #These are just filler for now
    testingImages_b = 0
    testingLabels = 0
    #y_b = addBias(testingImages)

    # Return the optimized weight matrix W
    W = softmaxRegression(X_b, y, testingImages_b, testingLabels, epsilon=0.1, batchSize=3623)
    
    np.save('W', W)

    # We can load the saved weights here if we don't want to go through the training process again.
    
    #loadedWeights = np.load("W.npy")
    #weights = np.delete(loadedWeights, (784), axis=0)
    """
    # Let's test the weights on the testing set
    optimized_FCE = calculateFCE(weights, testingImages, testingLabels)
    print("FCE on testing set -> ", optimized_FCE)
    newPC = calculatePC(weights, testingImages, testingLabels)
    print("PC on testing set -> ", newPC)

    print(weights.T.shape)
    w1 = weights.T[0].reshape(28,28)
    w2 = weights.T[1].reshape(28,28)
    w3 = weights.T[2].reshape(28,28)
    w4 = weights.T[3].reshape(28,28)
    w5 = weights.T[4].reshape(28,28)
    w6 = weights.T[5].reshape(28,28)
    w7 = weights.T[6].reshape(28,28)
    w8 = weights.T[7].reshape(28,28)
    w9 = weights.T[8].reshape(28,28)
    w10 = weights.T[9].reshape(28,28)





    # Visualize the vectors
    # We must remove the added constant term, and we can then visualize the weights.
    #plt.imshow(w1, cmap='gray'), plt.show()
    #plt.imshow(w2, cmap='gray'), plt.show()
    #plt.imshow(w3, cmap='gray'), plt.show()
    #plt.imshow(w4, cmap='gray'), plt.show()
    #plt.imshow(w5, cmap='gray'), plt.show()
    #plt.imshow(w6, cmap='gray'), plt.show()
    #plt.imshow(w7, cmap='gray'), plt.show()
    #plt.imshow(w8, cmap='gray'), plt.show()
    #plt.imshow(w9, cmap='gray'), plt.show()
    #plt.imshow(w10, cmap='gray'), plt.show()

    # Data augmentation (translation, rotation, scaling, and random noise)

    Xaug = np.empty([1, 784 ])
    print(trainingImages.shape)
    for x in range(trainingImages.shape[0]):
        choice = random.randint(0,3)

        if choice == 0:
            result = rotateImage(trainingImages[x])
        elif choice == 1:
            result = scaleImage(trainingImages[x])
        elif choice == 2:
            result = translateImage(trainingImages[x])
        else:
            result = randomNoise(trainingImages[x])
        
        Xaug = np.copy(np.append(Xaug, np.array([result.flatten()]), axis=0))
        print(x)
    #Remove that starting empty row
    Xaug = np.delete(Xaug, 0, 0)

    np.save('Xaug', Xaug)


    # Since the labels are the same, we can sinmply copy the original training labels
    # into our yaug

    yaug = np.copy(trainingLabels) 


    print(testingImages[0])
    result = rotateImage(testingImages[0])
    plt.imshow(result, cmap='gray'), plt.show()

    plt.imshow(trainingImages[0].reshape(28, 28), cmap='gray'), plt.show()
    scaled = scaleImage(testingImages[0])
    plt.imshow(scaled, cmap='gray'), plt.show()


    tranlated = translateImage(testingImages[0])
    plt.imshow(tranlated, cmap='gray'), plt.show()


    randomT = randomNoise(testingImages[0])
    plt.imshow(randomT, cmap='gray'), plt.show()
    
    """
    