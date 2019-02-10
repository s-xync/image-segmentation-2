import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from time import time


def getInputImageMatrix(imageName):
    # without giving a flag, imread will take in only
    # RGB channels and will leave transperancy out
    imageMatrix = cv2.imread(imageName)
    return imageMatrix.astype(float)


def saveOutputImage(clusterMatrix, noClusters, colorsList, imageName):
    height, width = clusterMatrix.shape
    outputImageMatrix = np.full((height, width, 3), 0)
    for i in range(height):
        for j in range(width):
            outputImageMatrix[i][j] = colorsList[clusterMatrix[i][j]]

    outputImageMatrix = outputImageMatrix.astype(np.uint8)
    cv2.imwrite(imageName, outputImageMatrix)
    return


def distanceBetweenPoints(featureVector1, featureVector2):
    # return np.sqrt(np.sum((featureVector1 - featureVector2)**2))
    return np.linalg.norm(featureVector1 - featureVector2)


def buildFeatureMatrix(
    inputImageMatrix,
    redColorFlag,
    greenColorFlag,
    blueColorFlag,
    xCordFlag,
    yCordFlag,
    textureFlag,
):
    start = time()
    depth = (
        (1 if redColorFlag else 0)
        + (1 if greenColorFlag else 0)
        + (1 if blueColorFlag else 0)
        + (1 if xCordFlag else 0)
        + (1 if yCordFlag else 0)
        + (1 if textureFlag else 0)
    )

    height, width, _ = inputImageMatrix.shape
    featureMatrix = np.zeros((height, width, depth))
    track_depth = 0
    if redColorFlag:
        featureMatrix[:, :, track_depth] = inputImageMatrix[:, :, 0] / 255 * 100
        track_depth += 1

    if greenColorFlag:
        featureMatrix[:, :, track_depth] = inputImageMatrix[:, :, 1] / 255 * 100
        track_depth += 1

    if blueColorFlag:
        featureMatrix[:, :, track_depth] = inputImageMatrix[:, :, 2] / 255 * 100
        track_depth += 1

    if xCordFlag:
        for i in range(0, height):
            for j in range(0, width):
                featureMatrix[i, j, track_depth] = j
        featureMatrix[:, :, track_depth] /= width
        track_depth += 1

    if yCordFlag:
        for i in range(0, height):
            for j in range(0, width):
                featureMatrix[i, j, track_depth] = i
        featureMatrix[:, :, track_depth] /= height
        track_depth += 1

    if textureFlag:
        greyImageMatrix = (
            inputImageMatrix[:, :, 0]
            + inputImageMatrix[:, :, 1]
            + inputImageMatrix[:, :, 2]
        ) / 3
        assert (
            greyImageMatrix.shape[0] == height and greyImageMatrix.shape[1] == width
        ), "Dimensions must be same."
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                featureMatrix[i, j, track_depth] = np.std(
                    np.array(
                        [
                            [
                                greyImageMatrix[i - 1, j - 1],
                                greyImageMatrix[i - 1, j],
                                greyImageMatrix[i - 1, j + 1],
                            ],
                            [
                                greyImageMatrix[i, j - 1],
                                greyImageMatrix[i, j],
                                greyImageMatrix[i, j + 1],
                            ],
                            [
                                greyImageMatrix[i + 1, j - 1],
                                greyImageMatrix[i + 1, j],
                                greyImageMatrix[i + 1, j + 1],
                            ],
                        ]
                    )
                )
        track_depth += 1

    # if textureFlag:
    #     # local binary pattern as texture
    #     greyImageMatrix = (
    #         inputImageMatrix[:, :, 0]
    #         + inputImageMatrix[:, :, 1]
    #         + inputImageMatrix[:, :, 2]
    #     ) / 3
    #     assert (
    #         greyImageMatrix.shape[0] == height and greyImageMatrix.shape[1] == width
    #     ), "Dimensions must be same."
    #     textureImageMatrix = local_binary_pattern(
    #         greyImageMatrix, 8, 1, method="uniform"
    #     )
    #     featureMatrix[:, :, track_depth] = textureImageMatrix
    #     track_depth += 1

    assert track_depth == depth, "Dimensions must be same."
    end = time()
    print("Feature matrix built in {0}s.".format(end - start))
    return featureMatrix
