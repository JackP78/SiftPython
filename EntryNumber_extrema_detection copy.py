import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve

def create_octave_images(init_img, s, sigma): 
  octave = [init_img] 
  k = 2**(1/s) 
  for i in range(s+2): 
    next_img = cv2.GaussianBlur(octave[-1], (0, 0), sigma)
    octave.append(next_img) 
  return octave

def create_pyramid(img, s, sigma): 
  pyramid = [] 
  for octave in range(4): 
    octave_images = create_octave_images(img, s, sigma) 
    pyramid.append(octave_images) 
    img = octave[-3][::2, ::2] 
  return pyramid

def compute_dog_octave(octave_in):
   octave = []
   for i in range(1, len(octave_in)):
      octave.append(octave_in[i] - octave_in[i-1])
   

def scale_space_extrema_detection2(img, num_octaves=4, num_scales=5, threshold=1):
    # Create an empty list to store extrema points
    extrema_points = []

    for octave in range(num_octaves):
        scale_factor = 2 ** octave
        for s in range(1, num_scales - 2):
            # Compute the scale
            scale = scale_factor * (2 ** (s / num_scales))

            # Apply Gaussian filter to the image
            blurred_img = cv2.GaussianBlur(img, (0, 0), sigmaX=scale, sigmaY=scale)

            # Find local extrema in the Difference of Gaussians (DoG)
            dog = img - blurred_img
            maxima = (dog > threshold) & (np.roll(dog, 1, axis=0) > threshold) & (np.roll(dog, -1, axis=0) > threshold) & \
                     (np.roll(dog, 1, axis=1) > threshold) & (np.roll(dog, -1, axis=1) > threshold)

            # Get coordinates of the local maxima
            keypoint_coords = np.argwhere(maxima)

            for coords in keypoint_coords:
                y, x = coords[0], coords[1]
                extrema_points.append((x, y, scale))

    return extrema_points
