# David McCormick - DTM190000

import numpy as np

def convolve2d(image, kernel):
    # Takes image which is a single channel NxN (grayscale) image where:
    #  N is the height and width of the image
    # Also takes kernel, which is OxPxQ where:
    #  O is the number of kernels
    #  P is the kernel height
    #  Q is the kernel width
    # Will return a convolved image, which is

    # Obtain values of the image's shape
    imageHeight, imageWidth = image.shape

    # Obtain values of kernel's shape
    kernelDepth, kernelHeight, kernelWidth = kernel.shape

    # Calculates the dimensions of the convolved image
    #  Formula would change if we accounted for padding
    outputHeight = int(imageHeight - kernelHeight + 1)

    # Initialize output vectors
    # The convolved image will be
    output = np.zeros((kernelDepth, outputHeight, outputHeight))
    C1S1 = np.zeros(output.shape)

    # Goes over each filter
    for n in range(kernelDepth):
        # Goes over the entire image, finding the dot product of the kernel and the image
        for i in range(outputHeight):
            for j in range(outputHeight):
                # The actual convolving of the image
                currDotProduct = np.sum(image[i:(i + kernelHeight), j:(j + kernelWidth)] * kernel[n])
                output[n, i, j] = currDotProduct

                if currDotProduct > 0:
                    C1S1[n, i, j] = 1
                else:
                    C1S1[n, i, j] = 0


    return output, C1S1

def convolve3d(images, kernel):
    # Takes images which is the result of multiple filters on a single image,
    #  which will be MxNxN (grayscale) image where:
    #  M is the number of convolved images
    #  N is the height and width of the images
    # Also takes kernel, which is OxPxQxR where:
    #  O is the number of PxQxR kernels
    #  P is the kernel depth
    #  Q is the kernel height
    #  R is the kernel width

    # Obtain values of the image's shape
    imagesDepth, imageHeight, imageWidth = images.shape

    # Obtain values of the kernel's shape
    numKernels, kernelDepth, kernelHeight, kernelWidth = kernel.shape


    # Calculates the dimensions of the convolved image
    #  Formula would change if we accounted for padding
    outputHeight = int((imageHeight - kernelHeight) + 1)

    # Initialize output vectors
    # The convolved image will be
    output = np.zeros((numKernels, outputHeight, outputHeight), dtype=np.float64)
    C2S2 = np.zeros(output.shape, dtype=np.float64)
    S2P1 = np.zeros(images.shape + output.shape, dtype=np.float64)

    # Goes over each filter
    for m in range(numKernels):
        # Goes over the entirety of the image
        for u in range(outputHeight):
            for v in range(outputHeight):
                # Accounts for the depth of the kernel
                currDotProduct = np.sum(images[0:kernelDepth, u:(u + kernelHeight), v:(v + kernelHeight)] * kernel[m])
                output[m, u, v] = currDotProduct

                if (currDotProduct > 0):
                    C2S2[m, u, v] = 1
                else:
                    C2S2[m, u, v] = 0
                S2P1[0:numKernels, u:(u + kernelHeight), v:(v + kernelHeight), m, u, v] = kernel[m]

    return output, C2S2, S2P1
