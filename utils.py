from cv2 import imread, imwrite
import numpy as np


def getInputImageMatrix(imageName):
    # without giving a flag, imread will take in only
    # RGB channels and will leave transperancy out
    imageMatrix = imread(imageName)
    return imageMatrix.astype(float)


def saveOutputImage(clusterMatrix, clusterCenters, imageName):
    height = clusterMatrix.shape[0]
    width = clusterMatrix.shape[1]
    outputImageMatrix = np.full((height, width, 3), 0)
    for i in range(height):
        for j in range(width):
            outputImageMatrix[i][j] = clusterCenters[clusterMatrix[i][j]]
    outputImageMatrix = outputImageMatrix.astype(np.uint8)
    imwrite(imageName, outputImageMatrix)
    return


def buildFeatureMatrix(inputImageMatrix, redColorFlag, greenColorFlag,
                       blueColorFlag, xCordFlag, yCordFlag, textureFlag):
    depth = ((1 if redColorFlag else 0) + (1 if greenColorFlag else 0) +
             (1 if blueColorFlag else 0) + (1 if xCordFlag else 0) +
             (1 if yCordFlag else 0) + (1 if textureFlag else 0))

    height = inputImageMatrix.shape[0]
    width = inputImageMatrix.shape[1]
    featureMatrix = np.zeros((height, width, depth))
    track_depth = 0
    if redColorFlag:
        featureMatrix[:, :, track_depth] = inputImageMatrix[:, :, 0]
        track_depth += 1

    if greenColorFlag:
        featureMatrix[:, :, track_depth] = inputImageMatrix[:, :, 1]
        track_depth += 1

    if blueColorFlag:
        featureMatrix[:, :, track_depth] = inputImageMatrix[:, :, 2]
        track_depth += 1

    if xCordFlag:
        for i in range(0, height):
            for j in range(0, width):
                featureMatrix[i, j, track_depth] = j
        track_depth += 1

    if yCordFlag:
        for i in range(0, height):
            for j in range(0, width):
                featureMatrix[i, j, track_depth] = i
        track_depth += 1

    if textureFlag:
        greyImageMatrix = (
            inputImageMatrix[:, :, 0] + inputImageMatrix[:, :, 1] +
            inputImageMatrix[:, :, 2]) / 3
        assert greyImageMatrix.shape[0] == height and greyImageMatrix.shape[
            1] == width, "Dimensions must be same."
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                featureMatrix[i, j, track_depth] = np.std(
                    np.array([[
                        greyImageMatrix[i - 1, j - 1],
                        greyImageMatrix[i - 1, j],
                        greyImageMatrix[i - 1, j + 1]
                    ],
                              [
                                  greyImageMatrix[i, j - 1],
                                  greyImageMatrix[i, j],
                                  greyImageMatrix[i, j + 1]
                              ],
                              [
                                  greyImageMatrix[i + 1, j - 1],
                                  greyImageMatrix[i + 1, j],
                                  greyImageMatrix[i + 1, j + 1]
                              ]]))
        track_depth += 1

    assert track_depth == depth, "Dimensions must be same."
    return featureMatrix
