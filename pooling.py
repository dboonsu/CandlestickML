import numpy as np

def avgPooling(image, stride=2):
    # Takes image which is a single channel MxNxN (grayscale) image where:
    #  M is the depth of the image, or number of channels
    #  N is the height and width of the image
    # Also takes stride, which is O where:
    #  O is how far the window will pool
    # Will return a smaller, pooled image

    # Obtain values of the image's shape
    imageDepth, imageHeight, imageWidth = image.shape

    # Calculate the size of the pooled image
    pooledHeight = int(imageHeight/stride)

    # Initialize output vectors
    # The convolved image will be
    output = np.zeros((imageDepth, pooledHeight, pooledHeight), dtype=(np.float64))
    indices = np.zeros((imageDepth, pooledHeight, pooledHeight), dtype=(np.int64, 2))

    # Goes over every channel
    for n in range(imageDepth):
        # Goes over the height and width of the image
        for i in range(pooledHeight):
            for j in range(pooledHeight):
                # Selects a window and calculate the mean value of the window
                region = image[n, (2 * i):(2 * i + 2), (2 * j):(2 * j + 2)]
                output[n, i, j] = np.mean(region)

                # You need to keep travel of the indices of the max value for the gradient
                indexI, indexJ = np.unravel_index(np.argmax(region), region.shape)
                indices[n, i, j] = [2 * i + indexI, 2 * j + indexJ]

    return output, indices

def maxPooling(image, stride=2):
    # Takes image which is a single channel MxNxN (grayscale) image where:
    #  M is the depth of the image, or number of channels
    #  N is the height and width of the image
    # Also takes stride, which is O where:
    #  O is how far the window will pool
    # Will return a smaller, pooled image

    # Obtain values of the image's shape
    imageDepth, imageHeight, imageWidth = image.shape

    # Calculate the size of the pooled image
    pooledHeight = int(imageHeight/stride)

    # Initialize output vectors
    # The convolved image will be
    output = np.zeros((imageDepth, pooledHeight, pooledHeight), dtype=(np.float64))
    indices = np.zeros((imageDepth, pooledHeight, pooledHeight), dtype=(np.int64, 2))

    # Goes over every channel
    for n in range(imageDepth):
        # Goes over the height and width of the image
        for i in range(pooledHeight):
            for j in range(pooledHeight):
                # Selects a window and finds the maximum value of the window

                # Selecting the window
                region = image[n, (2 * i):(2 * i + 2), (2 * j):(2 * j + 2)]

                # Finding the max
                output[n, i, j] = np.max(region)

                # Get the local indices of the local max of the region
                # You need to keep track of the indices of the max value for the gradient
                localX, localY = np.unravel_index(np.argmax(region), region.shape)

                # Must shift the indices of the entire image
                indices[n, i, j] = [2 * i + localX, 2 * j + localY]

    return output, indices