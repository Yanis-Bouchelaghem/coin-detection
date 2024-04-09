import cv2
import numpy as np
import os

CoinConfig = {
    "1 Euro": {
        "value": 1,
        "radius": 23.25,
        "ratio": 1.4307,
        "count": 0,
    },
    "2 Euro ": {
        "value": 2,
        "radius": 25.75,
        "ratio": 1.5846,
        "count": 0,
    },
    "1 Cent": {
        "value": 0.01,
        "radius": 16.25,
        "ratio": 1,
        "count": 0,
    },
    "2 Cent": {
        "value": 0.02,
        "radius": 18.75,
        "ratio": 1.12195,
        "count": 0,
    },
    "5 Cent": {
        "value": 0.05,
        "radius": 21.25,
        "ratio": 1.3076,
        "count": 0,
    },
    "10 Cent": {
        "value": 0.1,
        "radius": 19.75,
        "ratio": 1.2153,
        "count": 0,
    },
    "20 Cent": {
        "value": 0.2,
        "radius": 22.25,
        "ratio": 1.3692,
        "count": 0,
    },
    "50 Cent": {
        "value": 0.5,
        "radius": 24.25,
        "ratio": 1.4923,
        "count": 0,
    },
}

def detect_coins(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=30, minRadius=6,
                               maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Compare each circle with all others
        circles_to_remove = []
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                # Compute distance between centers
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # Check if circle j is inside circle i
                if distance <= r1:
                    circles_to_remove.append(j)
                elif distance <= r2:
                    circles_to_remove.append(i)
                    continue

        # Remove circles that are inside another circle
        circles = np.delete(circles, circles_to_remove, axis=0)

        # calculate_amount(circles)

        # Draw remaining circles
        for (x, y, r) in circles:
            # Draw the circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            # Draw the center of the circle
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

        # Display the result
        # cv2.imshow("Detected Coins", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return circles

def detect_coins_without_display(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=30, minRadius=6,
                               maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Compare each circle with all others
        circles_to_remove = []
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                # Compute distance between centers
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # Check if circle j is inside circle i
                if distance <= r1:
                    circles_to_remove.append(j)
                elif distance <= r2:
                    circles_to_remove.append(i)
                    continue

        # Remove circles that are inside another circle
        circles = np.delete(circles, circles_to_remove, axis=0)

        # Draw remaining circles
        for (x, y, r) in circles:
            # Draw the circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            # Draw the center of the circle
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

        # print(circles.shape[0])
        return circles.shape[0]

    return 0

def calculate_amount(mapped_image):
    circles = detect_coins(mapped_image)
    image = cv2.imread(mapped_image)
    mappedImage = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
    radius = []
    coordinates = []
    if circles is not None:
        for x, y, detected_radius in circles:
            radius.append(detected_radius)
            coordinates.append([x, y])

        smallest = min(radius)
        threshold = 0.05555555  # 0.035
        total_money = 0
        coins_on_board = 0
        font = cv2.FONT_HERSHEY_COMPLEX

        for circle in circles:
            ratio_to_check = circle[2] / smallest
            (x, y, r) = circle
            # Draw the circle
            cv2.circle(mappedImage, (x, y), r, (0, 255, 0), 4)
            # Draw the center of the circle
            cv2.circle(mappedImage, (x, y), 2, (0, 0, 255), 3)
            for rub in CoinConfig:
                if abs(ratio_to_check - CoinConfig[rub]['ratio']) <= threshold:
                    value = CoinConfig[rub]['value']
                    CoinConfig[rub]['count'] += 1
                    total_money += value
                    cv2.putText(mappedImage, str(value), (int(circle[0]) - 25, int(circle[1]) + 20), font, 1,
                                (0, 0, 0), 5)
                    coins_on_board += 1
                    break

        # cv2.imwrite("mappedMoney1.jpg", mappedImage)
        cv2.imshow('result', mappedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        height, width, channels = mappedImage.shape
        avg_color = mappedImage.mean(axis=0).mean(axis=0).round()

        print('\n height, width, avg_color, nombre de pieces, total')
        return height, width, avg_color, coins_on_board, total_money
    return 0, 0, 0, 0, 0


if __name__ == "__main__":
    # Example usage
    # detect_coins('data/IMG_1647.JPG')
    directory = 'data'
    for filename in os.listdir(directory):
        v = calculate_amount('data/'+filename)
        print(v[4])