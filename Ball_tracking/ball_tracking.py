import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


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

        l_1 = np.array([0, 200, 95])
        u_1 = np.array([10, 255, 255])

        l_2 = np.array([160, 100, 20])
        u_2 = np.array([179, 255, 255])

        lower_mask = cv2.inRange(hsv, l_1, u_1)
        upper_mask = cv2.inRange(hsv, l_2, u_2)

        # Mask generated for frame
        full_mask = lower_mask + upper_mask

        # Find the pixels that belong to the ball
        ball_pixels = np.nonzero(full_mask)

        # Check if there are any pixels that belong to the ball
        if ball_pixels[0].size != 0 and ball_pixels[1].size != 0:
            # Calculate the center of the ball
            cx = np.mean(ball_pixels[1])
            cy = np.mean(ball_pixels[0])

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

    x = np.linspace(min(center_points_x), max(center_points_x), 100)
    y = a*x**2 + (b*x) + c
    print(f"y = {a}*x**2 + {b}*x + {c}")
    return a, b, c, x, y


def main():

    All_images = ret_images("ball.mov")

    x_center, y_center = find_ball(All_images)

    a, b, c, x, y = fit_parabolic_curve(x_center, y_center)
    # print(a,b,c)

    landing_y = 300 + y_center[0]
    landing_x = (-b + np.sqrt(b**2 - 4*a*(c - landing_y)))/(2*a)
    print(f"Landing x-coordinate: {landing_x} pixels")

    data = cv2.cvtColor(All_images[55], cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 10))
    plt.plot(x, y, color="white", linewidth=2)
    plt.scatter(x_center, y_center, s=5)
    plt.plot(x, y, '-r')
    plt.annotate('Ball Trajectory', xy=(x_center[110]+10, y_center[110]),
                 fontsize=14, xytext=(900, 280),
                 arrowprops=dict(facecolor='blue'),
                 color='white')
    plt.annotate('LSF curve', xy=(x[30], y[30]-5),
                 fontsize=14, xytext=(200, 200),
                 arrowprops=dict(facecolor='red'),
                 color='white')
    plt.axis([min(x_center), max(x_center), max(y_center), 0])
    plt.axis('off')
    plt.imshow(data)
    plt.show()


if __name__ == "__main__":
    main()
