'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import reconstruction as sub
from helper import camera2

def reconstructPoints(im1, im2, coords1, coords2, K1, K2, pts1):
    # coords1 = np.int32(coords1)
    # coords2 = np.int32(coords2)
    F, mask = cv2.findFundamentalMat(coords1,coords2,cv2.FM_LMEDS)
    # from helper import epipolarMatchGUI
    # epipolarMatchGUI(im1, im2, F)
    coords1 = coords1[mask.ravel()==1]
    coords2 = coords2[mask.ravel()==1]

    E = sub.essentialMatrix(F, K1, K2)
    M1 = np.concatenate((np.eye(3), np.ones((3,1))), axis=1)
    C1 = np.dot(K1, M1)
    M2s = camera2(E)

    pts2 = np.zeros(pts1.shape)

    for i in range(pts1.shape[0]):
        x = pts1[i, 0]
        y = pts1[i, 1]
        x2,y2 = sub.epipolarCorrespondence(im1, im2, F, x, y)
        pts2[i,:] = [x2, y2]
    # for (x,y) in pts1:
    #     cv2.circle(im1,(x,y),3,255,-1)
    # plt.imshow(im1),plt.show()
    # for (x,y) in pts2:
    #     cv2.circle(im2,(np.int(x),np.int(y)),3,255,-1)
    # plt.imshow(im2),plt.show()

    for i in range(M2s.shape[2]):
        M2 = M2s[:,:,i]
        C2 = np.dot(K2, M2)
        P = cv2.triangulatePoints(C1, C2, pts1.T, pts2.T).T
        if np.min(P[:,2]) > 0:
            break

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax.set_xlim3d(-2.5, 1)
    # ax.set_ylim3d(-2.5, -0.25)
    # ax.set_zlim3d(-1000, 11.5)

    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return P