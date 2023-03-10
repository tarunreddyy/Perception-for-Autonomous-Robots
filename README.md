# Perception-for-Autonomous-Robots

This repository contains the code and data for the Perception for Autonomous Robots course (ENPM673) at the University of Maryland, College Park. The code is written in Python and uses various libraries such as NumPy, Matplotlib, and OpenCV.

## Projects

- Ball_tracking: This program uses computer vision techniques to track a moving ball and predict its trajectory. It takes a video file as input and returns a plot of the trajectory with a fitted parabolic curve.
- Camera_position_estimation: A computer vision program that detects an A4 piece of paper in a video feed and computes a homography matrix to determine the camera's position relative to the paper using world coordinates as a reference. The program's output is the camera's position, pitch, roll, and yaw.
- Image_stitching: Image stitching using feature detection, matching, RANSAC algorithm and Homography to obtain the final panorama image.
- This code provides three methods for fitting a plane to 3D point data: Standard least squares, Total least squares, RANSAC.


## Dependencies
- Python 3.7 or higher
- OpenCV
- NumPy
- Matplotlib
- tqdm

## Usage
- Clone the repository: git clone https://github.com/tarunreddyy/Perception-for-Autonomous-Robots.git
- Navigate to the project directory and follow the instructions in the README file for the project

## License
[MIT](https://choosealicense.com/licenses/mit/)
