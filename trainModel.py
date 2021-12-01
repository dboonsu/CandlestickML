# David McCormick - DTM190000

import pooling
import convolve
import numpy as np

def trainTestSplit(images, labels, percent):
    # Splits the images into training and testing based on the percent passed inx

    # Input validation
    if percent <= 0 or percent >= 1:
        print("Please input a proper train/test split percent")
        exit(0)

    # Gets the number of images that will be used for training
    maxImages = len(images)
    maxNumTrain = int(maxImages * percent)

    # Randomizes the order of the arrays
    permutation = np.random.permutation(len(images))
    images = images[permutation]
    labels = labels[permutation]

    # Train test split
    trainImages = images[0:maxNumTrain]
    trainLabels = labels[0:maxNumTrain]
    testImages = images[maxNumTrain:maxImages]
    testLabels = labels[maxNumTrain:maxImages]

    return trainImages, trainLabels, testImages, testLabels

def flatten(image):
    # Flattens the (M, N, O) image into an (MxNxO, ) array`
    imageDepth, imageHeight, imageWidth = image.shape
    flattened = np.reshape(image, (imageDepth * imageHeight * imageWidth,))
    return flattened

def softmax(weights, flattened):
    # Takes weights which is MxN
    #  M classes
    #  N weights per class
    # Also takes flattened
    #  The image we are currently looking at, flattened
    # Assigns probabilities that a given image is a class in a 1xN vector
    #   These probabilities will sum up to one
    x = np.dot(weights, flattened)
    predictions = np.exp(x) / np.sum(np.exp(x))
    return predictions

def crossEntropyLoss(probability):
    # Cross entropy loss, determines how sure we are that the prediction is right
    #  If the probability is 1, then loss is 0
    #  Loss will be larger as we are more unsure
    loss = -np.log(probability)
    return loss

def predict(image, kernel1, kernel2, weights):
    # Takes image, kernels, and weights
    #  Will perform the necessary steps to predict the class of the image
    #  Follows the convolutional neural network architecture we have set up

    # Convolution Layer One
    C1, C1S1 = convolve.convolve2d(image, kernel1)

    # ReLu Activation
    C1[C1 <= 0] = 0

    # MaxPool Layer One
    P1, I1 = pooling.maxPooling(C1)

    # Convolution Layer Two
    C2, C2S2, S2P1 = convolve.convolve3d(P1, kernel2)

    # ReLu Activation
    C1[C1 <= 0] = 0

    # Pooling Layer Two
    P2, I2 = pooling.maxPooling(C2)

    # Fully Connected Layer
    f = flatten(P2)

    # Returns the probabilities for each predicted class
    probabilities = softmax(weights, f)

    return probabilities

def train(trainImages, trainLabels, testImages, testLabels, numEpochs, learningRate, kernel1, kernel2, weights):

    kernel1Depth, kernel1Height, kernel1Width = kernel1.shape
    kernel2Number, kernel2Depth, kernel2Height, kernel2Width = kernel2.shape

    # Contains the number of epochs
    epochs = []

    # Contains the average losses per epoch
    avgLosses = []

    # Contains the accuracy of the model per epoch
    accuracies = []

    for epoch in range(numEpochs):
        # Shuffle the training data
        permutation = np.random.permutation(len(trainImages))
        trainImages = trainImages[permutation]
        trainLabels = trainLabels[permutation]
        # Zip makes trainImages and trainLabels iterable together
        for img, label in zip(trainImages, trainLabels):
            # Convolution Layer One
            C1, C1S1 = convolve.convolve2d(img, kernel1)

            # ReLu Activation
            C1[C1 <= 0] = 0

            # MaxPool Layer One
            P1, I1 = pooling.maxPooling(C1)
            pooled1Depth, pooled1Height, pooled1Width = P1.shape

            # Convolution Layer Two
            C2, C2S2, S2P1 = convolve.convolve3d(P1, kernel2)

            # ReLu Activation
            C2[C2 <= 0] = 0

            # MaxPool Layer Two
            P2, I2 = pooling.maxPooling(C2)
            pooled2Depth, pooled2Height, pooled2Width = P2.shape

            # Fully Connected Layer
            f = flatten(P2)
            probabilities = softmax(weights, f)

            # Backpropagation works from fully connected back up to C1
            # Calculate gradient for classes
            #  Just the probabilities for the predictions
            #  except you subtract one for the label
            LS = np.copy(probabilities)
            LS[label] = probabilities[label] - 1

            # Calculate gradient for weights
            #  The probabilities times the flattened vector
            #  except, the column of the label is the probability minus 1
            LW = np.zeros(probabilities.shape + f.shape)
            for i in range(probabilities.shape[0]):
                LW[i, :] = probabilities[i] * f
            LW[label, :] = (probabilities[label] - 1) * f

            # Calculate gradient for the pooling layer two
            #  Has the shape of P2, the second pooled layer
            LP2 = np.dot(LS, weights).reshape(P2.shape)

            # Used in the calculation of kernel2
            #  Backpropagation through the
            LC2 = np.zeros(C2.shape)
            for i in range(pooled2Depth):
                for j in range(pooled2Height):
                    for k in range(pooled2Width):
                        indexI, indexJ = I2[i, j, k]
                        LC2[i, indexI, indexJ] = LP2[i, j, k]

            # Used in the calculation for kernel2
            #  C2S2 is calculated during convolution 2
            #  Consists of 0's where the gradient of 0 or negative, 1's where the gradient is positive
            LS2 = LC2 * C2S2

            # Calculate the gradient for kernel2
            LK2 = np.zeros(kernel2.shape)
            for i in range(kernel2Number):
                for j in range(kernel2Depth):
                    for k in range(kernel2Height):
                        for l in range(kernel2Width):
                            LK2[i, j, k, l] = np.sum(LS2[i] * P1[j][k:(k + C2.shape[1]), l:(l + C2.shape[2])])

            # Used in gradient for kernel1
            #  Backpropagation through the pooling layer 1
            LP1 = np.zeros(P1.shape)
            for i in range(pooled1Depth):
                for j in range(pooled1Height):
                    for k in range(pooled1Width):
                        LP1[i, j, k] = np.sum(LS2 * S2P1[i, j, k])

            # Used in calculation for kernel1
            #  Backpropagation through the convolution layer 1
            LC1 = np.zeros(C1.shape)
            for i in range(pooled1Depth):
                for j in range(pooled1Height):
                    for k in range(pooled1Width):
                        indexI, indexJ = I1[i, j, k]
                        LC1[i, indexI, indexJ] = LP1[i, j, k]

            # Used in the calculation for kernel1
            #  C1S1 is calculated in the convolution layer one
            #  Consists of 1's where the gradient is above 0, and 0's where its equal to below 0
            LS1 = LC1 * C1S1

            # Calculate the gradient for kernel1
            LK1 = np.zeros(kernel1.shape)
            for i in range(kernel1Depth):
                for j in range(kernel1Height):
                    for k in range(kernel1Width):
                        LK1[i, j, k] = np.sum(LS1[i] * img[j:(j + C1.shape[1]), k:(k + C1.shape[2])])

            # Update the kernels
            kernel1 = kernel1 - learningRate * LK1
            kernel2 = kernel2 - learningRate * LK2

            # Update the weights
            weights = weights - learningRate * LW

        losses = np.zeros(1,)
        successes = 0

        # Iterates over the testing images and labels
        #  Uses the current kernel and weights to test model accuracy
        for img, label in zip(testImages, testLabels):
            # Calculates the probability that the image is of each class
            probabilities = predict(img, kernel1, kernel2, weights)

            # Gets the loss for the predicted value
            #  Index probabilities to the value of the label
            #  If the probability is 1, then the loss is zero
            currLoss = crossEntropyLoss(probabilities[label])
            losses = np.append(losses, currLoss)

            prediction = np.argmax(probabilities)
            if (prediction == label):
                successes += 1

        losses = np.delete(losses, 0)
        epochs.append(epoch + 1)

        # Calculates the average losses
        avgLosses.append(losses.mean())

        # Calculates the accuracy
        accuracy = round(100 * successes / len(testImages), 2)
        accuracies.append(accuracy)

        print("Epoch: " + str(epoch + 1) + "\tAverage Losses: " + str(losses.mean()) + "\tAccuracy: " + str(accuracy))

    # Once it has reached the max number of epochs
    return (epochs, avgLosses, accuracies, kernel1, kernel2, weights)
