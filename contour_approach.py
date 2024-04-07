import cv2
import numpy as np
import os

def detect(filename):


    image = cv2.imread(filename)
    image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayscale image", gray)

    blur = cv2.GaussianBlur(gray, (11,11), 0)
    cv2.imshow("Blurred image", blur)

    canny = cv2.Canny(blur, 30, 150, 3)
    cv2.imshow("Canny", canny)

    dilated = cv2.dilate(canny, (1,1), iterations = 10)
    cv2.imshow("Dilated", dilated)

    (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, contours, -1, (0,255,0), 2)

    cv2.imshow("contours on rgb",rgb)


    #print('Coins in the image: ', len(contours))

    # Filter contours based on area and shape
    min_area = 500
    max_area = 500020
    min_circularity = 0.6
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            print(filename+' perimeter is 0')
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if area > min_area and area < max_area and circularity > min_circularity:
            # Fit a circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)

            circles.append((int(x), int(y), int(radius)))

    # Draw the detected circles on the original image
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

    # Display the result
    cv2.imshow("final", image)
    print(len(circles))
    cv2.waitKey()
    cv2.destroyAllWindows()
    return len(circles)

def detect_without_display(filename):


    image = cv2.imread(filename)
    image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("grayscale image", gray)

    blur = cv2.GaussianBlur(gray, (11,11), 0)
    #cv2.imshow("Blurred image", blur)

    canny = cv2.Canny(blur, 30, 150, 3)
    #cv2.imshow("Canny", canny)

    dilated = cv2.dilate(canny, (1,1), iterations = 18)
    #cv2.imshow("Dilated", dilated)

    (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, contours, -1, (0,255,0), 2)

    #cv2.imshow("contours on rgb",rgb)


    #print('Coins in the image: ', len(contours))

    # Filter contours based on area and shape
    min_area = 500
    max_area = 500020
    min_circularity = 0.6
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            print(filename+' perimeter is 0')
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if area > min_area and area < max_area and circularity > min_circularity:
            # Fit a circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)

            circles.append((int(x), int(y), int(radius)))

    # Draw the detected circles on the original image
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

    # Display the result
    #cv2.imshow("final", image)
    print(len(circles))
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    return len(circles)

if __name__ == "__main__":
    detect('data/IMG_1647.JPG')
    directory = 'data'
    for filename in os.listdir(directory):
        detect('data/'+filename)
