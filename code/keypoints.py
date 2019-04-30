import numpy as np
import cv2
from matplotlib import pyplot as plt

def getCorrespondence(img1, img2):
    # Initiate SIFT detector
    orb = cv2.ORB_create()  

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)


    pts1, pts2 = [], []

    for i in range(20):
        pts1.append(kp1[matches[i].queryIdx].pt)
        pts2.append(kp2[matches[i].trainIdx].pt)

    pts1, pts2 = np.array(pts1), np.array(pts2)

    # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)

    # plt.imshow(img3),plt.show()

    corners = cv2.goodFeaturesToTrack(img1,200,0.01,10)
    corners = np.int0(corners)

    kp_reconstruct = []

    for i in corners:
        x,y = i.ravel()
        kp_reconstruct.append([x,y])
        # cv2.circle(img,(x,y),3,255,-1)

    kp_reconstruct = np.array(kp_reconstruct)

    return pts1, pts2, kp_reconstruct

    # plt.imshow(img),plt.show()