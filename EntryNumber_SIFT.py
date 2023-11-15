import cv2
from EntryNumber_extrema_detection import find_keypoints, calculate_difference_of_gaussian
from EntryNumber_Keypoint_Localization import localize_keypoints
from EntryNumber_Orientation_assignment import assign_keypoint_orientations


def perform_sift(img):
    dog_images = calculate_difference_of_gaussian(img, 4)
    print (f'lenght of dog_images: {len(dog_images)}')

    extrema_points = find_keypoints(dog_images)
    print(f'lenght of points after DOG: {len(extrema_points)}')
    # Display the original image with keypoints
    image_with_keypoints = cv2.drawKeypoints(img, [cv2.KeyPoint(x=0.5 * (2 ** octave) * j, y=0.5 * (2 ** octave) * i, _size=10) for octave, i in extrema_points for j in range(1, dog_images[octave].shape[1] - 1)], None)
    cv2.imshow("Image after DOG", image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    localized_keypoints = localize_keypoints(img, extrema_points)
    # Display the original image with localized keypoints
    print(f'lenght of points after localization: {len(localized_keypoints)}')
    image_with_localized_keypoints = img.copy()
    for x, y in localized_keypoints:
        cv2.circle(image_with_localized_keypoints, (int(x), int(y)), 5, (0, 255, 0), 2)
    cv2.imshow("After localization", image_with_localized_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Assign orientations to keypoints
    keypoint_orientations = assign_keypoint_orientations(img, localized_keypoints)
    print(f'lenght of points after orientation: {len(keypoint_orientations)}')

    return keypoint_orientations
