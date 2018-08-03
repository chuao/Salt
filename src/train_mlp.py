import pickle
file_X = open('./data/X.pkl','rb')
X = pickle.load(file_X)
file_X.close()
file_y = open('./data/y.pkl','rb')
y = pickle.load(file_y)
file_y.close()
imgs_files = open('./data/images.pkl','rb')
selected_files = pickle.load(imgs_files)
imgs_files.close()

from salt import *
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD



simple_model = run_mlp(X, y)
