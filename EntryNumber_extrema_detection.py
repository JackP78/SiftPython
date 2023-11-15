import cv2
import numpy as np


# this uses the build in function so I know it works, trying to recreate the logic myself in the following code
def built_in_sift(img):

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Generate DoG keypoints
    sift = cv2.SIFT_create()

    # Set the number of octaves
    num_octaves = 4

    # Detect key points and compute descriptors
    keypoints, _ = sift.detectAndCompute(gray_image, None)

    # Return the keypoints and descriptors
    return keypoints

# create the pyramids for each octave and return the images to be used for difference of guassian
def calculate_difference_of_gaussian(image, num_octaves=4):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Generate Gaussian pyramids
    pyramids = [gray_image]
    for _ in range(num_octaves):
        gray_image = cv2.pyrDown(gray_image)
        pyramids.append(gray_image)

    # Calculate the difference of guassian
    dog_images = [cv2.resize(pyramids[i], pyramids[i+1].shape[::-1]) - pyramids[i+1] for i in range(num_octaves)]

    return dog_images


# for each octave return the difference of guassian and local extrema
def find_keypoints(dog_images):
    keypoints = []

    for octave_index in range(len(dog_images)):
        for scale_index in range(1, dog_images[octave_index].shape[0] - 1):
            current_scale = dog_images[octave_index][scale_index]
            upper_scale = dog_images[octave_index][scale_index + 1]
            lower_scale = dog_images[octave_index][scale_index - 1]

            # Check for extrema
            is_maxima = np.all(current_scale > upper_scale) and np.all(current_scale > lower_scale)
            is_minima = np.all(current_scale < upper_scale) and np.all(current_scale < lower_scale)

            if is_maxima or is_minima:
                keypoints.append((octave_index, scale_index))

    return keypoints
