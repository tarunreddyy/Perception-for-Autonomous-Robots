import cv2
import numpy as np
import matplotlib.pyplot as plt


def ret_images(file):
    # Images list
    images = []
    # Load video
    cap = cv2.VideoCapture(file)
    # Loop through frames of video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to HSV color space
        images.append(frame)
    # Release the video
    cap.release()

    return images


def find_ball(images):

    # Create lists to store center points of ball
    center_points_x = []
    center_points_y = []

    for i in range(len(images)):
        hsv = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)

        l_1 = np.array([0, 200, 50])
        u_1 = np.array([10, 255, 255])

        l_2 = np.array([160, 100, 20])
        u_2 = np.array([179, 255, 255])

        lower_mask = cv2.inRange(hsv, l_1, u_1)
        upper_mask = cv2.inRange(hsv, l_2, u_2)

        full_mask = lower_mask + upper_mask
        contours, _ = cv2.findContours(
            full_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Extracting x and y coordinates from an image
        # Find the largest contour (assume this is the ball)
        max_area = 0
        largest_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                largest_contour = cnt

        # Check if largest contour is not None
        if largest_contour is not None:
            # Calculate center of contour
            M = cv2.moments(largest_contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            center_points_x.append(cx)
            center_points_y.append(cy)

    center_points_x = np.array(center_points_x)
    center_points_y = np.array(center_points_y)
    return center_points_x, center_points_y


def fit_parabolic_curve(center_points_x, center_points_y):
    # y = ax^2 + bx + c
    # Create design matrix
    A = np.column_stack(
        [center_points_x**2, center_points_x, np.ones(len(center_points_x))])
    ATA = np.dot(A.T, A)
    ATy = np.dot(A.T, center_points_y)

    # Solve for coefficients
    coeff = np.linalg.solve(ATA, ATy)
    a, b, c = coeff

    x = np.linspace(min(center_points_x), max(center_points_x), 10)
    y = a*x**2 + (b*x) + c

    return a, b, c, x, y


def main():

    All_images = ret_images("ball.mov")

    x_center, y_center = find_ball(All_images)

    a, b, c, x, y = fit_parabolic_curve(x_center, y_center)
    print(a, b, c)

    landing_y = 300 + y_center[0]
    landing_x = (-b + np.sqrt(b**2 - 4*a*(c - landing_y)))/(2*a)
    print(f"Landing x-coordinate: {landing_x} pixels")

    plt.scatter(x_center, y_center)
    plt.plot(x, y, '-r')
    plt.axis([min(x_center), max(x_center), max(y_center), 0])
    plt.xlabel("X Coordinate (pixels)")
    plt.ylabel("Y Coordinate (pixels)")
    plt.title("Center Points of Ball")
    plt.show()


if __name__ == "__main__":
    main()
