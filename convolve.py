import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import cv2

def convolve(path, kernel):

    image = cv2.imread(path)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    imageHeight, imageWidth = image.shape
    print(imageHeight)
    print(imageWidth)
    print("-----------")

    # kernel = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])#np.zeros((3, 3))

    # kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # kernel = np.random.rand(3,3)
    kernelHeight = kernel.shape[0]
    kernelWidth = kernel.shape[1]
    # kernelDepth = len(kernel[0][0])
    print(kernelHeight)
    print(kernelWidth)
    # print(kernelDepth)
    print("-----------")

    outputHeight = imageHeight - kernelHeight + 1
    outputWidth = imageWidth - kernelWidth + 1
    output = np.zeros((outputHeight, outputWidth))
    print(outputHeight)
    print(outputWidth)
    print("-----------")

    currHeight = 0
    currWidth = 0

    for y in range(outputHeight):
        for x in range(outputWidth):
            output[y, x] = (kernel * image[y: y + kernelHeight, x: x + kernelWidth]).sum()

    cv2.imwrite("IMGDIR/CDL3BLACKCROWS/1A.jpg", output)


    # print(image[0:kernelHeight, 0:kernelWidth])
    # print(temp[0:kernelHeight, 1:kernelWidth+1])
    # print(temp[0:kernelHeight, 2:kernelWidth+2])
    # print(outputHeight)
    # print(outputWidth)
