# David McCormick - DTM190000

import cv2
import numpy as np
import os
import pandas as pd
import sys

import talib

import acquireCandlestickCharts
import DIRHELP
import testModel
import trainModel

if __name__ == "__main__":
    # Important for printing out entire weights and kernels
    np.set_printoptions(threshold=sys.maxsize)

    # 1 if you want to clear all the files in IMGDIR and remake the IMGDIR
    if (0):
        DIRHELP.reset()

    # Whether or not you want to acquire candlestick charts
    if (0):
        acquireCandlestickCharts.acquire()

    # For testing trained models
    if (0):
        testModel.test()

    # Stores all of the images in Mx28x28
    #  M is the total number of images
    #  28x28 is the number of pixels in each image
    images=np.zeros((28,28))

    # Labels stores all of the labels for the images in 1xM
    labels=np.zeros((1))
    dir = "IMGDIR/"

    # These are necessary because some of the candlestick patterns
    #  such as Evening Star is MUCH more frequent than Abandoned Baby
    MNOIPC = 10 # Max Number of Images Per Class
    CNOI = 0 # Current Number of Images Per Class
    NF = 0 # Number of folders
    for folder in os.scandir(dir):
        print(folder.name)
        CNOI = 0
        NF += 1
        for file in os.scandir(folder):
            # If we have reached the max number of files in this folder, we will skip to the next one
            if MNOIPC == CNOI:
                break

            # Formats the current file into something readable for cv2
            path = "IMGDIR/" + folder.name + "/" + file.name

            # Reads the current image with cv2 as grayscale
            current = cv2.imread(path, 0)

            # Resizes the image to be 28x28
            resize = cv2.resize(current, (28, 28), interpolation=cv2.INTER_AREA)

            # Appends the current image onto the Mx28x28 array
            images = np.dstack((images, resize))

            # Appends the current label onto the Mx1 array
            labels = np.append(labels, folder.name)

            CNOI+=1

    # Tranposes the images from 28x28xM to Mx28x28
    images = np.transpose(images, (2, 0, 1))

    # Arrays were formatted with np.zeros, removes that single entry
    images = np.delete(images, 0, 0)
    labels = np.delete(labels, 0)

    # Turns the classes into unique numbers
    labels = pd.factorize(labels)[0]

    # Turns images into floats
    images = images/255 - .5

    # Train Test Split
    trainPercent = .8 # 80/20 split
    trainImages, trainLabels, testImages, testLabels = trainModel.trainTestSplit(images, labels, trainPercent)

    # Shows the dimensions of the training and testing arrays
    print(trainImages.shape)
    print(trainLabels.shape)
    print(testImages.shape)
    print(testLabels.shape)

    # Kernels for convolutions
    kernel1 = np.random.uniform(low=-.1, high=.1, size=(4, 3, 3))
    kernel2 = np.random.uniform(low=-.1, high=.1, size=(6, 4, 3, 3))

    # For this calculation, see the bottom
    weightDim = (int((int((int(((28 - kernel1.shape[1] + 1) / 2)) - kernel2.shape[2] + 1) / 2)) ** 2) * kernel2.shape[0])


    # Weights for the fully connected layer
    # Weights are MxN where:
    #  M is the number of classes
    #  N is the number of pixels in the final, flattened convolved/pooled image
    weights = np.random.uniform(low=-.1, high=.1, size=(NF, weightDim))

    numEpochs = 10
    learningRate = .005

    # Trains the model!
    epochs, avgLosses, accuracy, kernel1, kernel2, weights = trainModel.train(trainImages, trainLabels, testImages,
                                                                              testLabels, numEpochs, learningRate,
                                                                              kernel1, kernel2, weights)
    with (open("output.txt", "a")) as f:
        print(repr(kernel1), file=f)
        print("kernel1", file=f)
        print("-----------------", file=f)
        print(repr(kernel2), file=f)
        print("kernel2", file=f)
        print("-----------------", file=f)
        print(repr(weights), file=f)
        print("Weights", file=f)
        print("-----------------", file=f)
        print(avgLosses, file=f)
        print("avgLosses", file=f)
        print("-----------------", file=f)
        print(accuracy, file=f)
        print("accuracy", file=f)
        print("-----------------", file=f)
    exit(0)

    # Example of calculating the weight dimensions
    #
    # 28 - 5 height + 1 / 2
    # For kernel1 = 1, 5, 5
    #     kernel2 = 1, 1, 5, 5
    # N is calculated as follows:
    #  C1
    #  (original pixel height) - (kernel1 height) + 1
    #  28 - 5 + 1 = 24
    #
    #  P1
    #  C1 height / 2
    #  24 / 2 = 12
    #
    #  C2
    #  (P1 pixel height) - (kernel2 height) + 1
    #  12 - 5 + 1 = 8
    #
    #  P2
    #  (C2 height / 2)
    #  8 / 2 = 4
    #
    #  Thus, image size is 4x4
    #  Final height * Final width * Number of k2 kernels
    #  4 * 4 * 1 = 16