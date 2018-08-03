import skimage.io as imio
import numpy as np
np.random.seed(42)
import colorsys as color
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join

from salt import *

images_to_X_y(path='data/checked/',
                  percent=20,
                  filter_size=9,
                  save=True)
