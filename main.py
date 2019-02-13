from time import time
from os import path
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from random import randint


def main():
    tempVar = input("\nImage Filename: ")
    assert isinstance(tempVar, str), "Input should be a string."
    assert path.isfile(tempVar), "Input should be a valid filename."
    imageFilename = tempVar

    tempVar = input("\nNo. of Clusters: ")
    assert tempVar.isdigit(), "Input should be a positive integer."
    assert int(tempVar) > 0, "Input should be greater than zero."
    noClusters = int(tempVar)

    tempVar = input("\nNo. of Iterations: ")
    assert tempVar.isdigit(), "Input should be a positive integer."
    assert int(tempVar) > 0, "Input should be greater than zero."
    noIterations = int(tempVar)

    print("\nWhich features do you want in the feature vector?")

    tempVar = input("Red Color (y/n): ")
    assert tempVar == "y" or tempVar == "n", "Input shoud be either 'y' or 'n'."
    redColorFlag = True if tempVar == "y" else False

    tempVar = input("Green Color (y/n): ")
    assert tempVar == "y" or tempVar == "n", "Input shoud be either 'y' or 'n'."
    greenColorFlag = True if tempVar == "y" else False

    tempVar = input("Blue Color (y/n): ")
    assert tempVar == "y" or tempVar == "n", "Input shoud be either 'y' or 'n'."
    blueColorFlag = True if tempVar == "y" else False

    tempVar = input("X-Coordinate (y/n): ")
    assert tempVar == "y" or tempVar == "n", "Input shoud be either 'y' or 'n'."
    xCordFlag = True if tempVar == "y" else False

    tempVar = input("Y-Coordinate (y/n): ")
    assert tempVar == "y" or tempVar == "n", "Input shoud be either 'y' or 'n'."
    yCordFlag = True if tempVar == "y" else False

    tempVar = input("Texture (y/n): ")
    assert tempVar == "y" or tempVar == "n", "Input shoud be either 'y' or 'n'."
    textureFlag = True if tempVar == "y" else False

    start = time()
    inputImageMatrix = getInputImageMatrix(imageFilename)
    print("")
    print("Building feature matrix started.")
    featureMatrix = buildFeatureMatrix(
        inputImageMatrix,
        redColorFlag,
        greenColorFlag,
        blueColorFlag,
        xCordFlag,
        yCordFlag,
        textureFlag,
    )
    print("Building feature matrix completed.")
    print("Kmeans clustering started.")
    clusterMatrix = kmeans(featureMatrix, noClusters, noIterations, inputImageMatrix)
    print("Kmeans clustering completed.")
    saveOutputImage(clusterMatrix, noClusters, inputImageMatrix, "output_final.jpg")
    end = time()
    print(
        "\n{0} kmeans clustering iteration(s) completed in {1}s.".format(
            noIterations, round(end - start, 1)
        )
    )


def getInputImageMatrix(imageName):
    # without giving a flag, imread will take in only
    # RGB channels and will leave transperancy out
    imageMatrix = cv2.imread(imageName)
    return imageMatrix.astype(float)


def saveOutputImage(clusterMatrix, noClusters, inputImageMatrix, imageName):
    height, width = clusterMatrix.shape
    outputImageMatrix = np.full((height, width, 3), 0)
    colorsList = []
    eachClusterSize = []
    for i in range(noClusters):
        colorsList.append(np.array([0.0, 0.0, 0.0]))
        eachClusterSize.append(0)

    for i in range(height):
        for j in range(width):
            colorsList[clusterMatrix[i, j]] += inputImageMatrix[i, j]
            eachClusterSize[clusterMatrix[i, j]] += 1

    for i in range(noClusters):
        colorsList[i] /= eachClusterSize[i]

    for i in range(height):
        for j in range(width):
            outputImageMatrix[i, j] = colorsList[clusterMatrix[i, j]]

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
        featureMatrix[:, :, track_depth] = inputImageMatrix[:, :, 0] / 255 * 50
        track_depth += 1

    if greenColorFlag:
        featureMatrix[:, :, track_depth] = inputImageMatrix[:, :, 1] / 255 * 50
        track_depth += 1

    if blueColorFlag:
        featureMatrix[:, :, track_depth] = inputImageMatrix[:, :, 2] / 255 * 50
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


def kmeans(featureMatrix, noClusters, noIterations, inputImageMatrix):
    """
    puts together calculateClusterMatrix & calculateClusterCenters
    """
    height, width, _ = featureMatrix.shape
    clusterCenters = pickRandomClusterCentersAndColors(
        featureMatrix, noClusters, height, width
    )
    # clusterMatrix = calculateClusterMatrix(featureMatrix, clusterCenters, noClusters)
    for i in range(noIterations):
        start = time()
        clusterMatrix = calculateClusterMatrix(
            featureMatrix, clusterCenters, noClusters
        )
        clusterCenters = calculateClusterCenters(
            featureMatrix, clusterMatrix, noClusters
        )
        saveOutputImage(
            clusterMatrix, noClusters, inputImageMatrix, "output_" + str(i) + ".jpg"
        )
        end = time()
        print("{0} iteration completed in {1}s.".format(i + 1, round(end - start, 1)))

    return clusterMatrix


def pickRandomClusterCentersAndColors(featureMatrix, noClusters, height, width):
    clusterCenters = []
    for _ in range(noClusters):
        clusterCenters.append(featureMatrix[randint(0, height), randint(0, width), :])

    return clusterCenters


def calculateClusterMatrix(featureMatrix, clusterCenters, noClusters):
    """
    calculates a 2d matrix with index of cluster centers as values
    """
    height, width, _ = featureMatrix.shape
    clusterMatrix = np.full((height, width), -1)
    for i in range(height):
        for j in range(width):
            distances = []
            for k in range(noClusters):
                distances.append(
                    distanceBetweenPoints(featureMatrix[i, j, :], clusterCenters[k])
                )
            clusterMatrix[i, j] = distances.index(min(distances))

    return clusterMatrix


def calculateClusterCenters(featureMatrix, clusterMatrix, noClusters):
    """
    calculates cluster centers
    """
    height, width, depth = featureMatrix.shape
    clusterCenters = []
    noPixelsPerCluster = []
    for i in range(noClusters):
        clusterCenters.append(np.zeros((depth)))
        noPixelsPerCluster.append(0)

    for i in range(height):
        for j in range(width):
            clusterCenters[clusterMatrix[i, j]] += featureMatrix[i, j]
            noPixelsPerCluster[clusterMatrix[i, j]] += 1

    for i in range(noClusters):
        clusterCenters[i] /= noPixelsPerCluster[i]

    return clusterCenters


if __name__ == "__main__":
    main()
