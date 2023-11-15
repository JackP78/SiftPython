import cv2
import numpy as np

def convolution(image, kernel):
     # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding for the output
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    # zero pad the matrix prior to convolution to handle border
    output = np.zeros_like(image)

    # Perform convolution for the single grey scale image
    for i in range(padding_height, image_height - padding_height):
        for j in range(padding_width, image_width - padding_width):
            # apply the kernel
            roi = image[i - padding_height:i + padding_height + 1,
                    j - padding_width:j + padding_width + 1]

            # Apply the convolution operation
            output[i, j] = np.sum(roi * kernel)

    return output

def guassianfilter(image, sigma=1):
    kernel_size = 6 * sigma + 1  # Choose an appropriate kernel size based on sigma
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel_2d = gaussian_kernel * gaussian_kernel.T

    # Normalize the kernel
    gaussian_kernel_2d /= gaussian_kernel_2d.sum()

    # Perform convolution using the convolution function
    image = convolution(image, gaussian_kernel_2d)

    return image

def my_harris_corner(gray_img, threshold=0.01):
    window_size=3
    k=0.04

    # Compute image gradients using Sobel operators
    dx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)

    # Compute components of the Harris matrix
    dxx = dx * dx
    dyy = dy * dy
    dxy = dx * dy

    # Apply Gaussian smoothing to the components of the Harris matrix
    guass_kernel = cv2.getGaussianKernel(window_size, 0)

    dxx = convolution(dxx, guass_kernel)
    dxx = convolution(dyy, guass_kernel)
    dyy = convolution(dyy, guass_kernel)

    # Compute the determinant and trace of the Harris matrix
    determinant = dxx * dyy - dxy**2
    trace = dxx + dyy

    # Compute the Harris response
    R = determinant - k * (trace**2)

    # Find interesting corner points that are above the threshold
    corner_indices = np.argwhere(R > R.max() * threshold)

    return corner_indices