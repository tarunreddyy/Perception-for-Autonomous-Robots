import numpy as np
import matplotlib.pyplot as plt

# Define a function to fit a plane to a set of points using standard least squares
def fit_plane_sls(points):
    # Find the mean of the x, y, and z values
    mean_x = np.mean(points[:, 0])
    mean_y = np.mean(points[:, 1])
    mean_z = np.mean(points[:, 2])

    # Subtract the mean from each x, y, and z value
    centered_points = points - np.array([mean_x, mean_y, mean_z])

    # Compute the covariance matrix of the centered points
    cov_matrix = np.cov(centered_points.T)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

    # The normal vector is the eigenvector corresponding to the smallest eigenvalue
    normal = eig_vecs[:, np.argmin(eig_vals)]

    # A point on the plane can be found by using the mean of the x, y, and z values
    point = np.array([mean_x, mean_y, mean_z])
    # print(point)
    return normal, point

# Define a function to fit a plane to a set of points using Total least squares
def fit_plane_tls(points):
    # Calculate mean of points
    mean = np.mean(points, axis=0)
    # Center the points
    centered_points = points - mean

    # Calculate singular value decomposition of centered points
    U, S, VT = np.linalg.svd(centered_points)

    # The normal vector is the last column of VT
    normal = VT[-1, :]

    # The point on the plane is the mean of the points
    point = mean

    return normal, point


# Define a function to calculate the distance from a point to a plane
def point_to_plane_distance(point, normal, plane_point):
    return np.abs(np.dot(normal, point - plane_point)) / np.linalg.norm(normal)

# Define a function to fit a plane to a set of points using RANSAC
def fit_plane_ransac(points, num_iterations=1000, inlier_threshold=0.1):
    best_normal = None
    best_point = None
    max_inliers = 0

    # Repeat the fitting process multiple times
    for i in range(num_iterations):
        # Randomly select 3 points from the data
        sample_indices = np.random.choice(len(points), size=3, replace=False)
        sample_points = points[sample_indices]

        # Fit a plane to the sample points using standard least squares
        normal, point = fit_plane_sls(sample_points)

        # Calculate the distance from each point to the fitted plane
        distances = [point_to_plane_distance(point, normal, p) for p in points]

        # Find the number of inliers (points within the inlier threshold)
        inlier_count = sum([1 for d in distances if d < inlier_threshold])

        # Update the best-fit plane if the current plane has more inliers
        if inlier_count > max_inliers:
            best_normal = normal
            best_point = point
            max_inliers = inlier_count

    # Return the best-fit plane
    return best_normal, best_point

# Define a function to plot points and plane
def plot_surface(points, normal, point, plot_title):
    # Compute the distance from the plane to the origin
    d = -np.dot(normal.T, point)

    # Normalize the normal vector
    norm = np.linalg.norm(normal)
    unit_normal = normal / norm

    # Calculate the limits of the data in the x and y dimensions
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    # Create a mesh grid for the x and y dimensions
    # Define a plane grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                         np.linspace(y_min, y_max, 10))
    z = (-unit_normal[0] * xx - unit_normal[1] * yy - d) * 1. / unit_normal[2]

    # Plot the points and the plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(plot_title)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.plot_surface(xx, yy, z, alpha=0.5)
    plt.show()


def main():
    # Load the data from the file
    points = np.loadtxt('pc1.csv', delimiter=',')
    # print(points.shape)
    # Fit a plane to the data using standard least squares
    normal_sls, point_sls = fit_plane_sls(points)
    plot_surface(points, normal_sls, point_sls, "Least Square Fitting pc1.csv")

    # Fit a plane to the data using total least squares
    normal_tls, point_tls = fit_plane_tls(points)
    plot_surface(points, normal_tls, point_tls,
                 "Total Least Square Fitting pc1.csv")

    # Fit a plane to the data using RANSAC
    normal_ransac, point_ransac = fit_plane_ransac(points)
    plot_surface(points, normal_ransac, point_ransac, "RANSAC fitting pc1.csv")

    # Load the data from the file
    points = np.loadtxt('pc2.csv', delimiter=',')
    # print(points.shape)
    # Fit a plane to the data using standard least squares
    normal_sls, point_sls = fit_plane_sls(points)
    plot_surface(points, normal_sls, point_sls, "Least Square Fitting pc2.csv")

    # Fit a plane to the data using total least squares
    normal_tls, point_tls = fit_plane_tls(points)
    plot_surface(points, normal_tls, point_tls,
                 "Total Least Square Fitting pc2.csv")

    # Fit a plane to the data using RANSAC
    normal_ransac, point_ransac = fit_plane_ransac(points)
    plot_surface(points, normal_ransac, point_ransac, "RANSAC fitting pc2.csv")


if __name__ == "__main__":
    main()
