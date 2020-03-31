from __future__ import print_function, division

# files and directories
import os
from os import listdir
from os.path import isfile, join

# generate random numbers and selections
import random
from random import randint

# to use pytorch Dataset and DataLoader
import torch
from torch.utils.data  import Dataset, DataLoader


import scipy.ndimage # to use scipy.ndarray.rotate rotate an image by given angle and plane
from skimage import io, transform # read and write image and other image operation
import numpy as np # matrix computation and operation
import matplotlib.pyplot as plt # visualize and check results

# to use own modules
from data.paths_m import get_path_file

# load data by samples, when data sample is not too large
class read_image_data(Dataset):
    """Normal Image Dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
        :root_dir: Directory with all the images
        :transform (callable, optional): Optional transform to be applied on a sample
        """
        self.file_list = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.file_list[idx])


        image = np.array(io.imread(img_name))
        print(img_name, type(image))

        if self.transform:
            image = self.transform(image)

        return image

def transform(img):

    list_str = ["lrflip", "udflip", "translation", "rotate"]
    # list_str = ["lrflip", "udflip", "translation", "scale", "rotate", "shear"]

    pertubation = random.sample(list_str, k=2) # randomly select 2 pertubating manners
    # pertubation = random.choices(list_str, k=2) # repeatedly select - bootstrap
    print(pertubation)

    if "lrflip" in pertubation:
        if len(img.shape) in [2,3]:
            img[:, :, :] = img[:, ::-1, :]
        else:
            assert("image should be 2D or 3D")

    if "udflip" in pertubation:
        if len(img.shape) in [2, 3]:
            img[:, :, :] = img[::-1, :, :]
        else:
            assert ("image should be 2D or 3D")

    if "translation" in pertubation:
        x_shift = randint(3, round(img.shape[1]*0.05))
        y_shift = randint(3, round(img.shape[0]*0.05))
        if len(img.shape) in [2,3]:
            # shift along x-axis
            img = np.roll(img, x_shift, axis=1)
            # shift along y-axis
            img = np.roll(img, y_shift, axis=0)
        else:
            assert("image should be in 2D or 3D dimension")

    if "rotate" in pertubation:
        img = scipy.ndimage.rotate(img, random.random()*0.2*6.2832+0.01, (0,1))

    return img



dir_m, file_ex =  get_path_file()
print(dir_m, listdir(dir_m))
data2load = read_image_data(dir_m, transform=transform)





fig = plt.figure()

for i in range(len(data2load)):
    sample = data2load[i]

    print(i,sample.shape)

    ax = plt.subplot(1, 2, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample)

plt.show()
