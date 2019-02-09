from time import time
from os import path


def main():
    temp_var = input("\nImage Filename: ")
    assert isinstance(temp_var, str), "Input should be a string."
    assert path.isfile(temp_var), "Input should be a valid filename."
    image_filename = temp_var

    temp_var = input("\nNo. of Clusters: ")
    assert temp_var.isdigit(), "Input should be a positive integer."
    assert int(temp_var) > 0, "Input should be greater than zero."
    no_clusters = int(temp_var)

    temp_var = input("\nNo. of Iterations: ")
    assert temp_var.isdigit(), "Input should be a positive integer."
    assert int(temp_var) > 0, "Input should be greater than zero."
    no_iterations = int(temp_var)

    print("\nWhich features do you want in the feature vector?")

    temp_var = input("Red Color (y/n): ")
    assert temp_var == "y" or temp_var == "n", "Input shoud be either 'y' or 'n'."
    red_color = True if temp_var == "y" else False

    temp_var = input("Green Color (y/n): ")
    assert temp_var == "y" or temp_var == "n", "Input shoud be either 'y' or 'n'."
    green_color = True if temp_var == "y" else False

    temp_var = input("Blue Color (y/n): ")
    assert temp_var == "y" or temp_var == "n", "Input shoud be either 'y' or 'n'."
    blue_color = True if temp_var == "y" else False

    temp_var = input("X-Coordinate (y/n): ")
    assert temp_var == "y" or temp_var == "n", "Input shoud be either 'y' or 'n'."
    x_cord = True if temp_var == "y" else False

    temp_var = input("Y-Coordinate (y/n): ")
    assert temp_var == "y" or temp_var == "n", "Input shoud be either 'y' or 'n'."
    y_cord = True if temp_var == "y" else False

    temp_var = input("Texture (y/n): ")
    assert temp_var == "y" or temp_var == "n", "Input shoud be either 'y' or 'n'."
    texture_sd = True if temp_var == "y" else False

    start = time()
    end = time()
    print("Time taken is {0}s".format(round(end - start, 1)))


if __name__ == "__main__":
    main()
