import cv2
from matplotlib import pyplot as plt
import numpy as np
from EntryNumber_Harris import my_harris_corner
from EntryNumber_SIFT import perform_sift
from EntryNumber_extrema_detection import built_in_sift

image_path = '1.jpg'
img = cv2.imread(image_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners = my_harris_corner(gray_img)

print(corners)
# Draw red circles around the interesting corner points on the display image
for point in corners:
    x, y = point[1], point[0]
    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

# Display the image with the interesting corner points
img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_display)
plt.title('Harris Corner Detection (Manual)')
plt.show()

# for reference use the built in SIFT function as reference
# Detect key points and compute descriptors
keypoints = built_in_sift(img)

# Draw the keypoints on the original image
image_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

# Display the original image and the one with keypoints
cv2.imshow("Using built in function", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Now use my sift implementation, which doesn't work.  finds no keypoints
points = perform_sift(img)
print (points)
