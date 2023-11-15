import cv2
from matplotlib import pyplot as plt
import numpy as np
from EntryNumber_Harris import my_harris_corner
from EntryNumber_SIFT import perform_sift
from EntryNumber_extrema_detection import built_in_sift
from EntryNumber_extrema_detection import detect_keypoints_local_extrema

# https://github.com/JackP78/SiftPython

def show_points_on_img(image_path, corners, title):
    print(f"number of keypoints {len(corners)}")
    img = cv2.imread(image_path)

    # Draw red circles around the interesting corner points on the display image
    for point in corners:
        x, y = point[1], point[0]
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    # Display the image with the interesting corner points
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_display)
    plt.title(title)
    plt.show()

image_path = '1.jpg'
img = cv2.imread(image_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners = my_harris_corner(gray_img)
show_points_on_img('1.jpg', corners, 'Harris Corner Detection')

# I couldn't get the local extrema to return non empty list of points, I tried my best
local_extrema = detect_keypoints_local_extrema(gray_img)
show_points_on_img('1.jpg', local_extrema, 'Local Extrema implementation')

perform_sift(img)
