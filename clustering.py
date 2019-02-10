from random import randint
import numpy as np
from time import time
from utils import distanceBetweenPoints, saveOutputImage


def kmeans(featureMatrix, noClusters, noIterations):
    """
    puts together calculateClusterMatrix & calculateClusterCenters
    """
    height, width, _ = featureMatrix.shape
    clusterCenters, colorsList = pickRandomClusterCentersAndColors(
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
            clusterMatrix, noClusters, colorsList, "output_" + str(i) + ".jpg"
        )
        end = time()
        print("{0} iteration completed in {1}s.".format(i + 1, round(end - start, 1)))

    return clusterMatrix, colorsList


def pickRandomClusterCentersAndColors(featureMatrix, noClusters, height, width):
    clusterCenters = []
    for _ in range(noClusters):
        clusterCenters.append(featureMatrix[randint(0, height), randint(0, width), :])

    colorsList = []
    for i in range(noClusters):
        colorsList.append(list(np.random.choice(range(256), size=3)))

    return clusterCenters, colorsList


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
