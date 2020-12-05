import argparse

import matplotlib.image as Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-ksize', type=int, default=7, help='kernel size for blurring')
parser.add_argument('-position_threshold', type=int, default=1, help='position threshold')
parser.add_argument('-offroad_threshold', type=float, default=0.0, help='offroad threshold')
parser.add_argument('-smoother_threshold', type=float, default=0.0)
parser.add_argument('-smoother_kernel', type=int, default=0)
parser.add_argument('-map', type=str, default='i80')
opt = parser.parse_args()
kernel_size = (opt.ksize, opt.ksize)
trajectory_image = Image.imread(f'{opt.map}_{opt.ksize}g{opt.position_threshold}actrajectory_offroad.png')
stop_image = Image.imread(f'{opt.map}_stop_region.png')
new_image = np.zeros_like(trajectory_image)
new_image[:, :, 3] = 1
new_image[trajectory_image[:, :, 2] <= opt.offroad_threshold, 2] = 1
new_image[:, :, 0] = stop_image[:, :, 0]
new_image[:, :, 1] = stop_image[:, :, 1]
Image.imsave(f"{opt.map}_eval_region.png", new_image)
