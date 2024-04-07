import cv2
import numpy as np
import os


def detect_coins(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=30, minRadius=10,
                               maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # Draw the circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            # Draw the center of the circle
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

        print(circles.shape[0])
        return circles.shape[0]
    print(0)
    # Display the result
    cv2.imshow("Detected Coins", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

def detect_coins_without_display(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=30, minRadius=10,
                               maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # Draw the circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            # Draw the center of the circle
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

        print(circles.shape[0])
        return circles.shape[0]
    print(0)
    # Display the result
    #cv2.imshow("Detected Coins", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return 0




# Example usage
directory = 'data'
for filename in os.listdir(directory):
    detect_coins('data/'+filename)
