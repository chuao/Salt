# First the usual imports:

import skimage.io as imio
import numpy as np
np.random.seed(42)
import colorsys as color
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from os import listdir
from os.path import isfile, join

import pandas as pd


# ## Visual inspection of the data
#
# I write a function that mixes the seismic image and the salt mask
# in a way which allows for quickly human inspection of the two images together.
# And create a directory full of the combined images for manual inspection.

def read_mask(filename, path='data/masks/', normalize=True):

    mask_f_name = path + filename
    mask = imio.imread(mask_f_name)
    if normalize and mask.min() != mask.max():
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask

def read_seis(filename, path='data/images/', normalize=True):
    seis_f_name = path + filename
    seis = imio.imread(seis_f_name)
    seis = (seis[:,:,0]).reshape(seis.shape[0], seis.shape[1])
    if normalize and seis.min() != seis.max():
        seis = (seis - seis.min()) / (seis.max() - seis.min())
    return seis

def mask_seis(filename,
              mask_path='data/masks/',
              seis_path = 'data/images/',
              comb_path = 'data/combo/',
              save = False):
    import skimage.io as imio
    import numpy as np
    import colorsys as color
    from matplotlib import pyplot as plt

    mask = read_mask(filename).ravel()
    seis = read_seis(filename).ravel()

    vfunc = np.vectorize(color.hsv_to_rgb)
    comb = np.stack([mask, mask, seis], axis=1).reshape([101,101,3])
    comb = np.dstack(vfunc(comb[:, :, 0], comb[:, :, 1], comb[:, :, 2]))
    plt.imshow(comb)
    if save:
        plt.imsave(comb_path + filename, comb)

# Now we gather a list of all the mask filenames, with the hope that
# each of them have a corresponding seismic image.

def mask_dir(path ='data/masks/'):
    for filename in listdir(path):
        mask_seis(filename, save=True)

def rle(arr, inverse=False, rows=101):
    '''
    This function takes an one-indexed array of 1s and 0s and
    encodes the 1s as consecutive offset-length pairs.
    If inverse is True, it takes an array of offset-lenght pairs
    and decodes it as an array of 1s and 0s

    input: Array to encode/decode
           flag to determine encoding or decoding.
           number of rows if decoding (from 1d to 2d)
    output: Encoded or decoded array
    '''

    arr = np.asarray(arr)                  # force numpy
    n = len(arr)
    if inverse:
        arr = arr.reshape([arr.shape[0] // 2, 2])
        out_size = arr[-1].sum() - 1
        cols = out_size // rows
        out = np.zeros(out_size)
        for pair in arr:
            out[pair[0] -1:pair[0] + pair[1] - 1] = 1
        return out.reshape([rows, cols]).T
    else:
        y = np.array(arr[1:] != arr[:-1])       # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)       # must include last element posi
        z = np.diff(np.append(-1, i))           # run lengths
        p = np.cumsum(np.append(0, z))[:-1] +1  # positions
        v = arr[i]
        all_tog = np.array([p, z, v]).astype(int).T
        all_tog = all_tog[all_tog[:,2]==1][:,0:2]
        return all_tog.ravel()

def encode_all_masks(path ='data/masks/'):
    import re
    files = [f for f in listdir(path) if isfile(join(path, f))]
    out_file_name = 'my_train.txt'
    with open(out_file_name, 'w') as output_file:
        output_file.write('id,rle_mask\n')
        for f in files:
            mask = read_mask(f)
            submission = rle(mask.T.ravel())
            output = ','.join([f[:-4],
                               np.array2string(submission,
                                                precision=0,
                                                separator=' ',
                                                max_line_width=999999)])
            output = re.sub(',\[[\] ]*', ',', output)
            output = re.sub(' +', ' ', output)
            output = re.sub('\]', '', output)
            output = output_file.write(output + '\n')
    output_file.close()

def get_X_y(img_lbl_name, filter_size=5,       # images and masks/lables have the same names
               mask_path='data/masks/',      # but live in different directories
               seis_path = 'data/images/',
               get_y=True):
        '''
        function to build X and y from images and masks
        Input: image and masks names (list or array)
               maks path
               images path
        Ouput: X matrix and y vector
        '''
        # usual imports
        import skimage.io as imio
        import numpy as np
        from skimage.util import pad

        #Load the files:

        if get_y:
            mask = read_mask(img_lbl_name, path=mask_path)
        seis = read_seis(img_lbl_name, path=seis_path)

        filter_half = (filter_size - 1) // 2

        # capture shape
        rows = seis.shape[0]
        cols = seis.shape[1]

        # Pad image
        seis = pad(seis,
                   filter_half,
                   mode='reflect',
                   reflect_type='odd')
        # make y
        y = mask.ravel()

        for i in range(y.shape[0]):
            r0 = i // cols
            r1 = r0 + filter_size
            c0 = i % cols
            c1 = c0 + filter_size
            if i == 0:
                X = seis[r0:r1, c0:c1].ravel()
            else:
                X = np.vstack((X, seis[r0:r1, c0:c1].ravel()))

        if get_y:
            return X, y
        else:
            return X

def images_to_X_y(path='data/checked/',
                  percent=10,
                  filter_size=9,
                  save=True):
    '''
    This Function
    '''
    import datetime
    start = datetime.datetime.now()
    files = [f for f in listdir(path) if isfile(join(path, f))]
    # to use the entire dataset make percent = 100 :D
    size = int(len(files) * percent) // 100
    selected_files = np.random.choice(files, size, replace=False)
    first = True
    for f in selected_files:
        print(f)    # it is here for debugging
        if first:
            X, y = get_X_y(f, filter_size=filter_size)
            first = False
        else:
            X_tmp, y_tmp = get_X_y(f, filter_size=filter_size)
            X = np.vstack((X, X_tmp))
            y = np.concatenate((y, y_tmp))
            #    print(X.shape, y.shape)    # it is here for debugging

    end = datetime.datetime.now()
    print(end - start)

    if save:
        save_X_y(X, y, selected_files, path='data/')
    #return X, y, selected_files

def save_X_y(X, y, selected_files, path='data/'):
    '''
    This function saves X, y and the list of converted
    images to Pickle objects for faster loading

    Input: Directory where image names are to be
           saved.
    '''
    from os import listdir
    from os.path import isfile, join
    import pickle
    X_file_name = join(path,'X.pkl')
    while isfile(X_file_name):
        X_file_name = X_file_name[:-4] + '-1' + '.pkl'
    file_X = open(X_file_name,'wb')
    pickle.dump(X, file_X)
    file_X.close()


    y_file_name = join(path, 'y.pkl')
    while isfile(y_file_name):
        y_file_name = y_file_name[:-4] + '-1' + '.pkl'
    file_y = open(y_file_name,'wb')
    pickle.dump(y, file_y)
    file_y.close()

    imgs_file_name = join(path, 'images.pkl')
    while isfile(imgs_file_name):
        imgs_file_name = imgs_file_name[:-4] + '-1' + '.pkl'
    imgs_files = open(imgs_file_name,'wb')
    pickle.dump(selected_files, imgs_files)
    imgs_files.close()

def plot_roc(ytrue, yprob):
    '''
    Plots ROC curve for predictor.
    '''
    from sklearn.metrics import auc, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    fpr, tpr, threshold = roc_curve(ytrue, yprob)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def run_logistic(X, y, test_size=0.2, random_state=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import auc, roc_auc_score, roc_curve

    Xtrn, Xtst, ytrn, ytst = train_test_split(X,
                                              y,
                                              test_size=test_size,
                                              random_state=random_state)

    model = LogisticRegression(solver='sag', max_iter=1000)
    model.fit(Xtrn, ytrn)
    probs = model.predict_proba(Xtst)
    yprd = probs[:,1]
    plot_roc(ytst, yprd)
    return model

def define_nn_mlp_model(Xtrn, ytrn):
    ''' defines multi-layer-perceptron neural network '''
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.optimizers import SGD

    model = Sequential() # sequence of layers
    num_neurons_in_layer_1 = 81  # number of neurons in a layer (is it enough?)
    num_neurons_in_layer_2 = 49  # number of neurons in a layer (is it enough?)
    num_neurons_in_layer_3 = 25  # number of neurons in a layer (is it enough?)
    num_inputs = Xtrn.shape[1] # number of features
    num_classes = 1  # Salt or Not Salt
    model.add(Dense(units=num_neurons_in_layer_1, # First hidden layer same size as inputs
                    input_dim=num_inputs,
                    kernel_initializer='orthogonal',
                    activation='relu'))
    model.add(Dense(units=num_neurons_in_layer_2,
                    input_dim=num_neurons_in_layer_1,
                    kernel_initializer='orthogonal',
                    activation='sigmoid'))
    model.add(Dense(units=1, # it just has to predict Salt or not
                    input_dim=num_neurons_in_layer_2,
                    kernel_initializer='orthogonal',
                    activation='sigmoid')) # keep softmax as last layer
    sgd = SGD(lr=0.01, decay=1e-9, momentum=.5) # learning rate, weight decay, momentum; using stochastic gradient descent (keep)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"] ) # (keep)
    return model

def print_output(model, ytrn, ytst):
    '''prints model accuracy results'''
    ytrn_pred = model.predict_classes(Xtrn, verbose=0).ravel()
    ytst_pred = model.predict_classes(Xtst, verbose=0).ravel()
    train_acc = np.sum(ytrn == ytrn_pred, axis=0) / Xtrn.shape[0]
    print('\nTraining accuracy: %.2f%%' % (train_acc * 100))
    test_acc = np.sum(ytst == ytst_pred, axis=0) / Xtst.shape[0]
    print('Test accuracy: %.2f%%' % (test_acc * 100))

def run_mlp(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import auc, roc_auc_score, roc_curve

    Xtrn, Xtst, ytrn, ytst = train_test_split(X,
                                              y,
                                              test_size=test_size,
                                              random_state=random_state)
    model = define_nn_mlp_model(Xtrn, ytrn)
    model.fit(Xtrn, ytrn, epochs=25,
                      batch_size=200,
                      verbose=1,
                      validation_split=0.1)
    print_output(model, ytrn, ytst)



    yprob =  model.predict_proba(Xtst, verbose=0).ravel()
    plot_roc(yprob, ytrue)
    return model

def predict_mask(seis_file_name, model, filter_size):
    X = get_X_y(seis_file_name, filter_size, get_y=False)
    ypred = model.fit(X)
    return y.reshape(X.shape)
