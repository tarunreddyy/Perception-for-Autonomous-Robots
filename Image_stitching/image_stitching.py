import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_images(folder_path):
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                image_list.append(img)
    return image_list

def matcher(img1, img2, threshold=0.75):
    """
    Finds the best matches in the images
    input: Two grayscale images
    output: matches
    """
    # create SIFT detector object
    sift = cv2.SIFT_create()
    
    # detect keypoints and extract features from both images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # create brute-force matcher object
    bf = cv2.BFMatcher()

    # find 2 nearest neighbors for each feature in image 1
    matches = bf.knnMatch(des1, des2, k=2)

    # filter out matches that don't pass the threshold
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])

    # extract keypoints from good matches and create an array of their coordinates
    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    # convert the matches list to a numpy array for easier manipulation
    matches = np.array(matches)

    # draw matches on a new image and display it
    img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # return an array of coordinates for the good matches
    return matches

def get_homography(matches, iterations=1000, num_pts=4, thres=5):
    """
    Compute the homography matrix from the matches
    input: matches
    output: homography matrix
    """
    # Extract keypoints from matches
    kpts1 = matches[:, :2]
    kpts2 = matches[:, 2:]

    # Convert keypoints to numpy arrays which contain only the x,y point
    kpts1 = np.float32(kpts1)
    kpts2 = np.float32(kpts2)

    # Construct the two sets of points
    points1 = np.hstack((kpts1, np.ones((len(kpts1), 1))))
    points2 = np.hstack((kpts2, np.ones((len(kpts2), 1))))

    best_H = None
    best_count = 0

    for i in range(iterations):
        idx = np.random.choice(len(points2), num_pts, replace=False)
        sample1 = points1[idx, :]
        sample2 = points2[idx, :]

        # Calculate the homography between the sets of points
        pts2 = sample2
        pts1 = sample1

        n = len(pts2)
        A = np.zeros((2 * n, 9))
        for i in range(num_pts):
            A[i * 2, :] = [pts2[i, 0], pts2[i, 1], 1, 0, 0, 0, -pts2[i, 0] * pts1[i, 0], -pts2[i, 1] * pts1[i, 0], -pts1[i, 0]]
            A[i * 2 + 1, :] = [0, 0, 0, pts2[i, 0], pts2[i, 1], 1, -pts2[i, 0] * pts1[i, 1], -pts2[i, 1] * pts1[i, 1], -pts1[i, 1]]

        _, _, V = np.linalg.svd(A)
        H = V[-1, :].reshape((3, 3))
        H = H / H[2, 2]

        # Calculating inliers and counting them
        pt2_h = np.dot(H, points2.T)
        pt2_h = pt2_h / pt2_h[2, :]
        dist = np.linalg.norm(points1.T - pt2_h, axis=0)
        inliers = np.sum(dist < thres)

        if inliers > best_count:
            best_H = H
            best_count = inliers

    return best_H

def trim(frame):
    """
    Trims the black part of the image
    input: image
    output: trimmed image
    """
    nonzero_rows = np.any(frame, axis=1)
    nonzero_cols = np.any(frame, axis=0)
    min_row, max_row = np.where(nonzero_rows)[0][[0, -15]]
    min_col, max_col = np.where(nonzero_cols)[0][[0, -15]]
    return frame[min_row:max_row+1, min_col:max_col+1]



def pair_stitch(img1,img2,H):
    """
    Stitch two images
    input: two images
    output: stitched image
    """
    width = img2.shape[1] + img1.shape[1]

    height = max(img2.shape[0],img1.shape[0])

    stitched_img = cv2.warpPerspective(img2, H , (width,height))
    stitched_img[0:img1.shape[0],0:img1.shape[1]] = img1

    return stitched_img

def stitch_images(image_list):
    """
    Stitch all images
    input: image list
    output: panorama image
    """
    # Resize images
    resize_list = [cv2.resize(img,(int(img.shape[1]/5),int(img.shape[0]/5))) for img in image_list]
    # Convert images to grayscale
    gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in resize_list]

    matches12 = matcher(gray_list[0],gray_list[1])
    H12 = get_homography(matches12)
    img12_stitch = pair_stitch(resize_list[0],resize_list[1],H12)
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(img12_stitch, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    matches23 = matcher(gray_list[1],gray_list[2])
    H23 = get_homography(matches23)
    img23_stitch = pair_stitch(resize_list[1],resize_list[2],H23)
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(img23_stitch, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    matches34 = matcher(gray_list[2],gray_list[3])
    H34 = get_homography(matches34)
    img34_stitch = pair_stitch(resize_list[2],resize_list[3],H34)
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(img34_stitch, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    img12_stitch = trim(img12_stitch)
    img34_stitch = trim(img34_stitch)
    panorama = pair_stitch(img12_stitch,img34_stitch,np.dot(H23,H12))
    panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    return panorama

if __name__ == "__main__":

    Images = read_images("resources/")

    panarama_img = stitch_images(Images)
    plt.figure(figsize=(10,10))
    plt.imshow(panarama_img)
    plt.axis('off')
    plt.show()