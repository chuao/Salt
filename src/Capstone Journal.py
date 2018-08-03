
# coding: utf-8

# # Pass the Salt

# First the usual imports:

# In[1]:


import skimage.io as imio
import numpy as np
np.random.seed(42)
import colorsys as color
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from os import listdir
from os.path import isfile, join

import pandas as pd


# ## Visual inspection of the data
# 
# I write a function that mixes the seismic image and the salt mask in a way which allows for quickly human inspection of the two images together. And create a directory full of the combined images for manual inspection.

# In[2]:


def read_mask(filename, path='../data/masks/', normalize=True):
    
    mask_f_name = path + filename
    mask = imio.imread(mask_f_name)
    if normalize and mask.min() != mask.max():
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask

def read_seis(filename, path='../data/images/', normalize=True):
    seis_f_name = path + filename
    seis = imio.imread(seis_f_name)
    seis = (seis[:,:,0]).reshape(seis.shape[0], seis.shape[1])
    if normalize and seis.min() != seis.max():
        seis = (seis - seis.min()) / (seis.max() - seis.min())
    return seis
    
    
def mask_seis(filename, 
              mask_path='../data/masks/',
              seis_path = '../data/images/',
              comb_path = '../data/combo/',
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


# Quick test witha single image, first we plot the seismic, then the mask and then the combined image.

# In[3]:


mask = read_mask('0a1742c740.png')
plt.imshow(mask, cmap='gray');


# In[4]:


seis = read_seis('0a1742c740.png')
plt.imshow(seis, cmap='gray');


# In[5]:


mask_seis('0a1742c740.png')


# Now we gather a list of all the mask filenames, with the hope that each of them have a corresponding seismic image.

# In[6]:


path = '../data/masks/'
files = [f for f in listdir(path) if isfile(join(path, f))]

# uncomment below to run this again
#for filename in files:
#    mask_seis(filename, save=True)


# In[7]:


files[0][:-4]


# ## Important note:
# 
# The file `707f14c59a.png` seemed to have the mask horizontally flipped and inverted (0s vs 1s), so I decided to "fix it", now the image and the mask make a better match.
# 
# UPDATE: The flipping ended up in a different flavor of PNG which doesn't behave like the others, so I decided to discard it instead.
# 
# **BASICALLY ignore the next few cells**

# In[8]:


filename = '707f14c59a.png'


# In[9]:


seis = imio.imread('../data/images/'+ filename)
seis = (seis[:,:,0] / 255)
plt.imshow(seis, cmap='gray');


# In[12]:


mask = imio.imread('../data/masks/' + filename )
mask = (mask / 65535)
plt.imshow(mask, cmap='gray');


# In[13]:


# mask_path='../data/masks/'
# seis_path = '../data/images/'
# comb_path = '../data/combo/'
# import skimage.io as imio
# import numpy as np
# import colorsys as color
# from matplotlib import pyplot as plt

# mask_f_name = mask_path + filename
# seis_f_name = seis_path + filename
# mask = imio.imread(mask_f_name)
# mask = (mask / 255).reshape(mask.shape[0] * mask.shape[1])
# seis = imio.imread(seis_f_name)
# seis = (seis[:,:,0] / 255).reshape(seis.shape[0] * seis.shape[1])

# mask


# In[14]:


# plt.imshow(mask.reshape([101,101]))


# In[15]:


# vfunc = np.vectorize(color.hsv_to_rgb)
# comb = np.stack([mask, mask, seis], axis=1).reshape([101,101,3])
# comb = np.dstack(vfunc(comb[:, :, 0], comb[:, :, 1], comb[:, :, 2]))
# plt.imshow(comb)


# In[16]:


# plt.imsave(comb_path + filename, comb)


# ## Data cleanup
# 
# Some of the masks as PNGs were obviously wrong, so I manually looked at all of them (4000) and marked as 'suspicious' de ones with clear problems, the better the training data the better the result should be.
# 
# Next I will check if the Mask images match the 'train.csv' file to do that I will need a run-length encoder and a decoder.

# In[17]:


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


# Let's test it with a 'random' image `40dcff68b3.png`
# 
# First we look at it:

# In[18]:


filename = '40dcff68b3.png'
mask = read_mask(filename)
plt.imshow(mask, cmap='gray');


# Now we decode it and compare with the data from train.csv
#         
#     40dcff68b3,3536 4 3637 20 3738 36 3839 53 3940 70 4041 86 4142 99 4243 5959
# 

# In[23]:


submission = rle(mask.T.ravel())
print(submission)


# Now we re-encode it!

# In[24]:


plt.imshow(rle(submission, inverse=True, rows=101), cmap='gray');


# And finally we compare all of the mask images with their encoded versions in `train.csv`

# In[ ]:


import re
path = '../data/masks/'
files = [f for f in listdir(path) if isfile(join(path, f))]
out_file_name = '../my_train.txt'
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


# Which we did outside of Python (for simplicity)
# 
# ```Bash
# chuao@elrond:~/Galvanize/Capstone_candidates/Salt - - - - - - - - - - - - - - - - - - - -[17:49:44]
# $ sort data/train.csv >1.txt; sort my_train.txt > 2.txt; diff 1.txt 2.txt 
# chuao@elrond:~/Galvanize/Capstone_candidates/Salt - - - - - - - - - - - - - - - - - - - -[17:49:45]
# $
# ```
# 
# 
# **The Training file matches all of the mask images!**

# ## First Models
# 
# We'll select a subsample of the training data to train a Logistic Regression model and/or a very basic kinda deep MLP model.
# 
# To do that I will train the model to predict one pixel from 9, 25 or 49 input pixels. and for that I nned to build X and y from the images and the masks, padding the masks according to the size of the input.

# ## Padding tests:
# 
# In order for the analysis to ve 'less wrong' on the border pixels, I decided to compare the differen options for padding

# Let's look a a single image before padding.

# In[ ]:


plt.imshow(seis, cmap='gray')


# And let's compare it with different padded ones

# In[ ]:


filter_size = 99
from skimage.util import pad
plt.imshow(pad(seis, 7, mode='symmetric', reflect_type='odd'), cmap='gray')


# In[ ]:


filter_size = 99
from skimage.util import pad
plt.imshow(pad(seis, 7, mode='symmetric', reflect_type='even'), cmap='gray')


# In[ ]:


filter_size = 99
from skimage.util import pad
plt.imshow(pad(seis, 7, mode='reflect', reflect_type='odd'), cmap='gray')


# In[ ]:


filter_size = 99
from skimage.util import pad
plt.imshow(pad(seis, 7, mode='reflect', reflect_type='even'), cmap='gray')


# ## Convert the images/masks data into X and y
# 
# 
# 

# The followingfunction producess a matrix of 'number of pixels rows' x 'number of pixels used as features' columns, per file. Given that each file has 10201 pixels. Our X grows 10201 rows for each file included in it.

# In[25]:


def get_X_y(img_lbl_name, filter_size=5,       # images and masks/lables have the same names 
               mask_path='../data/masks/',      # but live in different directories
               seis_path = '../data/images/',):                                          
        '''
        function to build X and y from images and masks
        Input: image and masks names (list or array)
        Ouput: Xmatrix and y vector
        '''
        # usual imports
        import skimage.io as imio
        import numpy as np
        from skimage.util import pad
        
        #Load the files:
        
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

        return X, y


# Now we need a to do this for all of the files

# In[ ]:


import datetime
start = datetime.datetime.now()

path = '../data/checked/'
files = [f for f in listdir(path) if isfile(join(path, f))]

# comment to use the entire dataset
# or make percent = 100 :D
percent = 0.1
size = int(len(files) * percent) // 100
active_files = np.random.choice(files, size, replace=False)
filter_size = 3

first = True
for f in active_files:
    print(f)
    if first:
        X, y = get_X_y(f, filter_size=filter_size)
        first = False
    else:
        X_tmp, y_tmp = get_X_y(f, filter_size=filter_size)
        X = np.vstack((X, X_tmp))
        y = np.concatenate((y, y_tmp))
print(X.shape, y.shape)

end = datetime.datetime.now()
print(end - start)


# ### Making sure, it is actually correct!

# In[ ]:


plt.imshow(y[0:10201].reshape([101,101]), cmap='gray')


# In[ ]:


fig1 = plt.figure(figsize=[12,12]) # create a figure with the default size 

ax1 = fig1.add_subplot(111) 
ax1.imshow((X[10201:20402].reshape([101,101, 9])[:,:,8]), interpolation='none', cmap='gray')
ax1.set_title('ac6289352d.png')


# In[ ]:


plt.imshow(read_seis('7582c0b2e9.png'), cmap='gray')


# In[ ]:


plt.imshow(read_mask('8457263314.png'), cmap='gray')


# ### And it is!!!

# It took 7 min and 50 sec to transcribe 80 images with a 9x9 filter. SO I let it running all night for all of the images and save the pickle 
# 
# **UPDATE:** I lost the pickle! and Iam using a smaller subset for now

# In[ ]:


import datetime
start = datetime.datetime.now()

path = '../data/checked/'
files = [f for f in listdir(path) if isfile(join(path, f))]

# comment to use the entire dataset
# or make percent = 100 :D
percent = 1
size = int(len(files) * percent) // 100
active_files = np.random.choice(files, size, replace=False)
filter_size = 9

first = True
for f in active_files:
    print(f)
    if first:
        X, y = get_X_y(f, filter_size=filter_size)
        first = False
    else:
        X_tmp, y_tmp = get_X_y(f, filter_size=filter_size)
        X = np.vstack((X, X_tmp))
        y = np.concatenate((y, y_tmp))
print(X.shape, y.shape)

end = datetime.datetime.now()
print(end - start)


import pickle
file_X = open('../data/X.pkl','wb')
pickle.dump(X, file_X)
file_X.close()
file_y = open('../data/y.pkl','wb')
pickle.dump(y, file_y)
file_y.close()
file_files = open('../data/files.pkl','wb')
pickle.dump(active_files, file_files)
file_files.close()


# **WOW...** 10% of the files produced a y of 30MB and a X of 2.4GB, in 39 minutes!
# 
# Let's look at it. From the pickle objects, because it takes for ever to extract X and y from the images.

# In[26]:


import pickle
file_X = open('../data/X.pkl','rb')
X = pickle.load(file_X)
file_X.close()
file_y = open('../data/y.pkl','rb')
y = pickle.load(file_y)
file_y.close()
file_files = open('../data/files.pkl','rb')
active_files = pickle.load(file_files)
file_files.close()


# In[27]:


# first a histogram, because, histograms
plt.hist(X.ravel(), bins=100, density=True);


# In[28]:


# and for y too, I know... is just 1s and 0s,
# it tels me about class balance
plt.hist(y, bins=2, density=True);


# In[29]:


print(X.shape, y.shape)
print(np.count_nonzero(np.isnan(X)))
print(np.count_nonzero(np.isnan(y)))


# ## Simple model 1: Logistic Regression
# 
# Now that we have X and y, let's try a logistic regression to see how it goes.

# In[30]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_auc_score, roc_curve


# In[31]:


Xtrn, Xtst, ytrn, ytst = train_test_split(X, 
                                          y,
                                          test_size=0.33,
                                          random_state=42)


# In[32]:


print(Xtrn.shape, Xtst.shape, ytrn.shape, ytst.shape)


# In[33]:


# I keep the fitting process in its own cell
# so I don't need to run it again and again
simple_model_1 = LogisticRegression(solver='sag', max_iter=1000)


# In[34]:


simple_model_1.fit(Xtrn, ytrn)


# In[37]:


# calculate the fpr and tpr for all thresholds of the classification
probs = simple_model_1.predict_proba(Xtst)
yprd = probs[:,1]
fpr, tpr, threshold = roc_curve(ytst, yprd)
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Most likely this problem breaks the assumptions of a linear model, let's attack the non linearity with a simple Multi Layer Perceptron

# ## A neural net.... at last.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])


# In[67]:


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

def define_nn_mlp_model(Xtrn, ytrn):
    ''' defines multi-layer-perceptron neural network '''
    model = Sequential() # sequence of layers
    num_neurons_in_layer_1 = 81  # number of neurons in a layer (is it enough?)
    num_neurons_in_layer_2 = 49  # number of neurons in a layer (is it enough?)
    num_neurons_in_layer_3 = 25  # number of neurons in a layer (is it enough?)
    num_inputs = Xtrn.shape[1] # number of features 
    num_classes = 1  # number of classes, 0-9 (keep)
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


# In[68]:


simple_model_2 = define_nn_mlp_model(Xtrn, ytrn)
simple_model_2.fit(Xtrn, ytrn, epochs=25, 
                  batch_size=200, 
                  verbose=1,
                  validation_split=0.1)
    
print_output(simple_model_2, ytrn, ytst)


# In[71]:


ytst_pred_prob, ytst_pred


# In[72]:


# calculate the fpr and tpr for all thresholds of the classification
yprd =  simple_model_2.predict_proba(Xtst, verbose=0).ravel()


fpr, tpr, threshold = roc_curve(ytst, yprd)
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# At last! a worthy opponent!
# 
# ![No Alt text needed](../img/finally-a-worthy-opponent.jpg)

#  

# ## Enter CNNs (not the News Channel)

# ## To Augment or not to augment...
# 
# Are ~4000 images enough training data? I need to answer this question, sooner than later in the project.

# # Parking Lot

# <img src="https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />
# 

# ## How to run it in the cloud (in parallel)
# 
# https://cloud.google.com/ml-engine/docs/tensorflow/distributed-tensorflow-mnist-cloud-datalab
