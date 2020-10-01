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
parser.add_argument('-speed_map', type=bool, default=False)
opt = parser.parse_args()
kernel_size = (opt.ksize, opt.ksize)
trajectory_image = Image.imread(f'{opt.position_threshold}actrajectory.png')
trajectory_image[:, :, 0] = cv2.GaussianBlur(trajectory_image[:, :, 0], kernel_size, opt.ksize / 3)
trajectory_image[:, :, 1] = cv2.GaussianBlur(trajectory_image[:, :, 1], kernel_size, opt.ksize / 3)
trajectory_image[:, :, 2] = cv2.GaussianBlur(trajectory_image[:, :, 2], kernel_size, opt.ksize / 3)
if opt.offroad_threshold != 0.0:
    trajectory_image[:, :, 2] = np.min(np.stack([trajectory_image[:, :, 2], np.ones_like(trajectory_image[:, :, 2]) *
                                                 opt.offroad_threshold], axis=-1), axis=-1) / opt.offroad_threshold
    if opt.smoother_kernel is not 0:
        index = trajectory_image[:, :, 2] <= opt.smoother_threshold
        temp_map = trajectory_image[:, :, 2].copy()
        temp_map[temp_map > 0.95] = 0
        temp_map = cv2.GaussianBlur(temp_map, (opt.smoother_kernel, opt.smoother_kernel),
                                    opt.smoother_kernel / 3)
        max_value = opt.smoother_threshold
        normalized_map = np.min(np.stack([temp_map, np.ones_like(temp_map) * max_value], axis=-1), axis=-1)
        trajectory_image[index, 2] = normalized_map[index]

    Image.imsave(f"{opt.ksize}g{opt.position_threshold}actrajectory_offroad.png", trajectory_image)
else:
    Image.imsave(f"{opt.ksize}g{opt.position_threshold}actrajectory.png", trajectory_image)

if opt.speed_map:
    trajectory_image = Image.imread(f'speed.png')
    trajectory_image[:, :, 0] = cv2.GaussianBlur(trajectory_image[:, :, 0], kernel_size, opt.ksize / 3)
    trajectory_image[:, :, 1] = cv2.GaussianBlur(trajectory_image[:, :, 1], kernel_size, opt.ksize / 3)
    trajectory_image[:, :, 2] = cv2.GaussianBlur(trajectory_image[:, :, 2], kernel_size, opt.ksize / 3)
    Image.imsave(f"{opt.ksize}gspeed.png", trajectory_image)