import cv2
import matplotlib.image as Image
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-ksize', type=int, default=7, help='kernel size for blurring')
parser.add_argument('-position_threshold', type=int, default=1, help='position threshold')
parser.add_argument('-offroad_threshold', type=float, default=0.0, help='offroad threshold')
parser.add_argument('-smoother_threshold', type=float, default=0.0)
parser.add_argument('-smoother_kernel', type=int, default=0)
opt = parser.parse_args()
kernel_size = (opt.ksize, opt.ksize)
trajectory_image = Image.imread(f'{opt.position_threshold}actrajectory.png')
lane_image=np.zeros_like(trajectory_image)
lane_image[:, :, 0] = cv2.GaussianBlur(trajectory_image[:, :, 0], (7,7), 7 / 3)
lane_image[:, :, 1] = cv2.GaussianBlur(trajectory_image[:, :, 1], (7,7), 7 / 3)
lane_image[:, :, 2] = cv2.GaussianBlur(trajectory_image[:, :, 2], (7,7), 7 / 3)
trajectory_image[:, :, 0] = cv2.GaussianBlur(trajectory_image[:, :, 0], kernel_size, opt.ksize / 3)
trajectory_image[:, :, 1] = cv2.GaussianBlur(trajectory_image[:, :, 1], kernel_size, opt.ksize / 3)
trajectory_image[:, :, 2] = cv2.GaussianBlur(trajectory_image[:, :, 2], kernel_size, opt.ksize / 3)
if opt.offroad_threshold != 0.0:
    new_image = np.zeros_like(trajectory_image)
    new_image[:,:,3] = 1
    new_image[trajectory_image[:, :, 2] <= opt.offroad_threshold,2] = 1
    #new_image[lane_image[:, :, 2] > opt.offroad_threshold, 1] = 1
    Image.imsave(f"offroad_region.png", new_image)
else:
    Image.imsave(f"{opt.ksize}g{opt.position_threshold}actrajectory.png", trajectory_image)
