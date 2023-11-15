import cv2
import numpy as np

def assign_keypoint_orientations(image, keypoints, patch_size=15, num_bins=36):
    keypoint_orientations = []

    for x, y in keypoints:
        # Extract the region around the keypoint
        half_size = patch_size // 2
        patch = image[int(y - half_size):int(y + half_size + 1), int(x - half_size):int(x + half_size + 1)]

        # Compute gradients
        dx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude and orientation
        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees

        # Create a histogram of gradient orientations
        histogram, bin_edges = np.histogram(orientation, bins=num_bins, range=(-180, 180))

        # Find the bin with the highest count
        dominant_bin = np.argmax(histogram)

        # Calculate the dominant orientation
        dominant_orientation = bin_edges[dominant_bin] + 180 / num_bins

        keypoint_orientations.append((x, y, dominant_orientation))

    return keypoint_orientations


