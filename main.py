from time import time
from os import path
from utils import getInputImageMatrix, saveOutputImage, buildFeatureMatrix
from clustering import kmeans


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
    clusterMatrix, colorsList = kmeans(featureMatrix, noClusters, noIterations)
    print("Kmeans clustering completed.")
    saveOutputImage(clusterMatrix, noClusters, colorsList, "output_final.jpg")
    end = time()
    print(
        "\n{0} kmeans clustering iteration(s) completed in {1}s.".format(
            noIterations, round(end - start, 1)
        )
    )


if __name__ == "__main__":
    main()
