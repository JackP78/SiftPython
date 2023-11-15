import cv2
import numpy as np

def localize_keypoints(image, keypoints, threshold=0.03):
    localized_keypoints = []

    for octave_index, scale_index in keypoints:
        scale = 0.5 * (2 ** octave_index)
        x, y = scale * scale_index, scale * scale_index

        # Extract the region around the keypoint
        patch_size = 5  # Adjust the patch size as needed
        patch = image[int(y - patch_size):int(y + patch_size + 1), int(x - patch_size):int(x + patch_size + 1)]

        # Compute gradients
        Ix = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

        # Compute Harris corner response
        A = np.sum(Ix ** 2)
        B = np.sum(Iy ** 2)
        C = np.sum(Ix * Iy)
        det = A * B - C ** 2
        trace = A + B
        harris_response = det - 0.04 * trace ** 2

        # Apply threshold and add to localized keypoints
        if harris_response > threshold:
            localized_keypoints.append((x, y))

    return localized_keypoints
