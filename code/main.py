from visualize import reconstructPoints
from keypoints import getCorrespondence
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

K = np.asarray([3310.400000, 0.000000, 316.730000, 0.000000, 3325.500000, 200.550000, 0.000000, 0.000000, 1.000000])
K = K.reshape((3,3))

dataset = "../data/dino/dino%04d.png"
data_start = 1
data_end = 363

im0 = cv2.imread(dataset%(1), 0)
P = np.asarray([0,0,0])
for i in range(1,363):
    im1 = cv2.imread(dataset%(i+1), 0)
    pts1, pts2, kps = getCorrespondence(im0, im1)
    P0 = reconstructPoints(im0, im1, pts1, pts2, K, K, kps)
    im0 = im1
    P = np.vstack((P, P0))

P = P[1:, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.set_xlim3d(-2.5, 1)
# ax.set_ylim3d(-2.5, -0.25)
# ax.set_zlim3d(8, 11.5)

ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()