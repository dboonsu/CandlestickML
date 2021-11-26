import math

import numpy as np
import cv2

def maxPooling(path, size = 2, stride = 2):

    image = cv2.imread(path)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    imageHeight, imageWidth = image.shape

    outputHeight = ((imageHeight - size) / stride) + 1
    outputWidth = ((imageWidth - size) / stride) + 1


    outputHeight = math.ceil(outputHeight)
    outputWidth = math.ceil(outputWidth)

    output = np.zeros((outputHeight, outputWidth))

    print(imageHeight)
    print(imageWidth)
    print(outputHeight)
    print(outputWidth)

    for y in range(0, imageHeight, size):
        for x in range(0, imageWidth, size):
            output[int(y/size), int(x/size)] = np.max(image[y: y + size, x: x + size])

    print(output)

    cv2.imwrite("IMGDIR/CDL3BLACKCROWS/1B.jpg", output)