import cv2
import matplotlib.image as Image
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-ksize', type=int, default=7, help='kernel size for blurring')
parser.add_argument('-position_threshold', type=int, default=1, help='position threshold')
parser.add_argument('-offroad_threshold', type=float, default=0.0, help='offroad threshold')
opt = parser.parse_args()
kernel_size = (opt.ksize, opt.ksize)
trajectory_image = Image.imread(f'{opt.position_threshold}actrajectory.jpg')/255.
trajectory_image[:, :, 0] = cv2.GaussianBlur(trajectory_image[:, :, 0], kernel_size, opt.ksize/3)
trajectory_image[:, :, 1] = cv2.GaussianBlur(trajectory_image[:, :, 1], kernel_size, opt.ksize/3)
trajectory_image[:, :, 2] = cv2.GaussianBlur(trajectory_image[:, :, 2], kernel_size, opt.ksize/3)
if opt.offroad_threshold!=0:
    trajectory_image[:, :, 2]=np.min(np.stack([trajectory_image[:,:,2],np.ones_like(trajectory_image[:,:,2])*0.3],axis=-1),axis=-1)/0.3
Image.imsave(f"{opt.ksize}g{opt.position_threshold}actrajectory.jpg", trajectory_image)