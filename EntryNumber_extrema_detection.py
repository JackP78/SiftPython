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

def detect_keypoints_local_extrema(grey_image, num_octaves = 5, num_scales = 4, sigma = 1.6):

    keypoints = []

    for octave in range(num_octaves):
        for scale in range(num_scales):
            # Calculate the standard deviations of the Gaussian kernels for the current scale
            sigma_cur = sigma * 2 ** (scale / num_scales)
            # Apply Gaussian blur with the current sigma
            blurred = cv2.GaussianBlur(grey_image, (0, 0), sigmaX=sigma_cur)
            print(f'octave {octave} scale {scale}')

            # Find local extrema in the DoG image
            if scale > 0:
                dog = prev_blurred - blurred  # Use the previous blurred image to find maxima
                local_maxima = np.zeros_like(dog, dtype=bool)
                local_minima = np.zeros_like(dog, dtype=bool)

                # Iterate over the interior pixels to compare with their neighbors
                for i in range(1, dog.shape[0] - 1):
                    for j in range(1, dog.shape[1] - 1):
                        neighbors = dog[i-1:i+2, j-1:j+2]
                        assert len(neighbors == 26)

                        # Check if the center pixel is greater than its neighbors
                        local_maxima[i, j] = np.all(dog[i, j] > neighbors)
                        # Check if the center pixel is less than its neighbors
                        local_minima[i, j] = np.all(dog[i, j] < neighbors)

                # Get the coordinates of anywhere the local max and min is true
                print(f'max={len(local_maxima)} min=len{len(local_minima)}')

                maxima_coords = np.where(local_maxima)
                minima_coords = np.where(local_minima)

                print(f'2 max={len(maxima_coords)} min=len{len(minima_coords)}')

                # the pionts I find are getting lost in this step, I don't know why, I need to column stack in order to add them to the flattened array
                maxima_coords_2 = np.column_stack(maxima_coords)
                minima_coords_2 = np.column_stack(minima_coords )

                print(f'3 max={len(maxima_coords_2)} min=len{len(minima_coords_2)}')
                
                # Add keypoints to the list
                keypoints.extend(maxima_coords_2)
                keypoints.extend(minima_coords_2)

            # Update the previous blurred image for the next iteration
            prev_blurred = blurred

        # Resize the image for the next octave
        grey_image = cv2.resize(grey_image, (grey_image.shape[1] // 2, grey_image.shape[0] // 2))

    return keypoints

