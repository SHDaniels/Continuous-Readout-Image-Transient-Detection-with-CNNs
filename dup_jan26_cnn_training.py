#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""


copied from cnn_training_morestamps, now with more classes and more stamps !

here we are retraining the same architecture of cnn with my new stamps
[the stamps go down to a 1.01x brightness increase now]

mostly copied from best_3_outbook_cnn:
 This notebook will train another 3-output nn, this time using some kind of
dynamic learning rate and also saving the best epoch instead of the last one
"""


# In[ ]:


# In[18]:


# imports
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from astropy.io import fits
import scipy.ndimage as snm
from tqdm.auto import tqdm
import os
from pylab import cm
import matplotlib as mpl
import sys


save_name = sys.argv[1]


plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gist_gray"


# In[ ]:


# In[3]:


fpath_streak = "/lustre/lrspec/ZTF_data/training_aug24/stamps_no_trans/images/"


for i, file in enumerate(os.listdir(fpath_streak)):
    image_filename = os.fsdecode(file)
    image_filename_full = fpath_streak + image_filename
    print(image_filename)
    streak = np.load(image_filename_full)
    streak_arr = np.array(streak)
    print(np.shape(streak))
    # all_trans_stamps.append(trans_arr)
    if i == 0:
        all_streak_stamps = streak_arr
    else:
        all_streak_stamps = np.concatenate((all_streak_stamps, streak_arr), axis=0)


# In[4]:


np.shape(all_streak_stamps)


# In[5]:


fpath_trans = "/lustre/lrspec/ZTF_data/training_aug24/stamps_pos_trans_2/"

# all_trans_stamps =[] #np.zeros((len(os.listdir(fpath_trans)), 10, 20, 20))

image_path = fpath_trans + "images/"
info_path = fpath_trans + "info/"

for i, file in enumerate(os.listdir(image_path)):
    image_filename = os.fsdecode(file)
    image_filename_full = image_path + image_filename
    print(image_filename)
    info_filename = image_filename[:-4] + "_info.npy"
    info_filename_full = info_path + info_filename
    trans = np.load(image_filename_full)
    trans_arr = np.array(trans)
    infos = np.load(info_filename_full)
    infos_arr = np.array(infos)
    print(np.shape(trans))
    print(np.shape(infos))
    # all_trans_stamps.append(trans_arr)
    if i == 0:
        all_pos_stamps = trans_arr
        all_pos_stamps_info = infos_arr
    else:
        all_pos_stamps = np.concatenate((all_pos_stamps, trans_arr), axis=0)
        all_pos_stamps_info = np.concatenate((all_pos_stamps_info, infos_arr), axis=0)


# In[6]:


np.shape(all_pos_stamps)


# In[7]:


np.shape(all_pos_stamps)


# In[8]:


fpath_cr = "/lustre/lrspec/ZTF_data/training_aug24/stamps_cr/"

# all_trans_stamps =[] #np.zeros((len(os.listdir(fpath_trans)), 10, 20, 20))

for i, file in enumerate(os.listdir(fpath_cr)):
    image_filename = os.fsdecode(file)
    image_filename_full = fpath_cr + image_filename
    print(image_filename)
    trans = np.load(image_filename_full)
    trans_arr = np.array(trans)
    print(np.shape(trans))
    # all_trans_stamps.append(trans_arr)
    if i == 0:
        all_cr_stamps = trans
    else:
        all_cr_stamps = np.concatenate((all_cr_stamps, trans_arr), axis=0)


# In[9]:


np.shape(all_cr_stamps)


# In[10]:


fpath_trans = "/lustre/lrspec/ZTF_data/training_aug24/stamps_neg_trans_2/"

# all_trans_stamps =[] #np.zeros((len(os.listdir(fpath_trans)), 10, 20, 20))

image_path = fpath_trans + "images/"
info_path = fpath_trans + "info/"

for i, file in enumerate(os.listdir(image_path)):
    image_filename = os.fsdecode(file)
    image_filename_full = image_path + image_filename
    print(image_filename)
    info_filename = image_filename[:-4] + "_info.npy"
    info_filename_full = info_path + info_filename
    trans = np.load(image_filename_full)
    trans_arr = np.array(trans)
    infos = np.load(info_filename_full)
    infos_arr = np.array(infos)
    print(np.shape(trans))
    print(np.shape(infos))
    # all_trans_stamps.append(trans_arr)
    if i == 0:
        all_neg_stamps = trans_arr
        all_neg_stamps_info = infos_arr
    else:
        all_neg_stamps = np.concatenate((all_neg_stamps, trans_arr), axis=0)
        all_neg_stamps_info = np.concatenate((all_neg_stamps_info, infos_arr), axis=0)


# In[11]:


np.shape(all_neg_stamps)


# In[12]:


fpath_bg = "/lustre/lrspec/ZTF_data/training_aug24/stamps_background/images/"

# all_trans_stamps =[] #np.zeros((len(os.listdir(fpath_trans)), 10, 20, 20))

for i, file in enumerate(os.listdir(fpath_bg)):
    image_filename = os.fsdecode(file)
    image_filename_full = fpath_bg + image_filename
    print(image_filename)
    bg = np.load(image_filename_full)
    bg_arr = np.array(bg)
    print(np.shape(bg))
    # all_trans_stamps.append(trans_arr)
    if i == 0:
        all_bg_stamps = bg
    else:
        all_bg_stamps = np.concatenate((all_bg_stamps, bg_arr), axis=0)

    # all_trans_stamps[i] = trans
# neg_stamps = all_bg_stamps


# In[13]:


np.shape(all_bg_stamps)


# In[14]:


fig, ax = plt.subplots(10, 10)
for i in range(10):
    for h in range(10):
        # print(i, h)
        ax[i, h].plot(np.sum(all_pos_stamps[i * 10 + h], axis=1))
        # ax[i, h].imshow(all_neg_stamps[i*10 + h])
        ax[i, h].axis("off")
fig.show()


# In[16]:


all_pos_stamps_info[:, 0]


# In[19]:


# Combine all features
feat = np.concatenate(
    (all_bg_stamps, all_pos_stamps, all_neg_stamps, all_cr_stamps, all_streak_stamps),
    axis=0,
)


# Split features into training and testing sets once
feat_tr, feat_te = train_test_split(feat, random_state=302)

# Reshape features for the CNN
feat_tr = feat_tr.reshape(feat_tr.shape[0], feat_tr.shape[1], feat_tr.shape[2], 1)
feat_te = feat_te.reshape(feat_te.shape[0], feat_te.shape[1], feat_te.shape[2], 1)

# Process bness and lens separately after the split
bness = np.concatenate(
    (
        np.full(len(all_bg_stamps), 0),
        all_pos_stamps_info[:, 0],
        all_neg_stamps_info[:, 0],
        np.full(len(all_cr_stamps), 0),
        np.full(len(all_streak_stamps), 0),
    )
)

lens = np.concatenate(
    (
        np.full(len(all_bg_stamps), 0),
        all_pos_stamps_info[:, 1],
        all_neg_stamps_info[:, 1],
        np.full(len(all_cr_stamps), 0),
        np.full(len(all_streak_stamps), 0),
    )
)

targ = np.concatenate(
    (
        np.full(len(all_bg_stamps), 0),
        np.full(len(all_pos_stamps), 1),
        np.full(len(all_neg_stamps), 2),
        np.full(len(all_cr_stamps), 3),
        np.full(len(all_streak_stamps), 4),
    )
)

# Split targets and additional variables
bness_tr, bness_te = train_test_split(bness, random_state=302)
lens_tr, lens_te = train_test_split(lens, random_state=302)
targ_tr, targ_te = train_test_split(targ, random_state=302)

# One-hot encode the target variable
targ_tr_one_hot = to_categorical(targ_tr, num_classes=5)
targ_te_one_hot = to_categorical(targ_te, num_classes=5)


# In[20]:


targ_tr_one_hot


# In[22]:


lens_tr


# In[25]:


from sklearn.model_selection import KFold
import numpy as np


# adaptive learning rate
def adapt_learning_rate(epoch):
    if epoch < 25:
        return 0.001
    elif epoch < 50:
        return 0.0005
    elif epoch < 80:
        return 0.0001
    elif epoch < 110:
        return 0.00005
    elif epoch < 150:
        return 0.00001
    else:
        return 0.000005
    # let's just test that out and see how it is


# note: default Adam learning rate is 0.001

my_lr_scheduler = keras.callbacks.LearningRateScheduler(adapt_learning_rate)

# early stopping callback
earlystopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0.001, patience=40, restore_best_weights=True
)
# This callback will stop the training when there is no improvement in
# the loss for 30 consecutive epochs.
# also will restore the best weights instead of the most recent

wid = 10

# -- build a simple CNN
layer1 = keras.layers.Conv2D(
    16,
    kernel_size=(5, 5),
    padding="same",
    activation="relu",
    input_shape=(2 * wid, 2 * wid, 1),
    kernel_regularizer=keras.regularizers.l2(0.05),
)
layer2 = keras.layers.MaxPool2D(2)  # changed from 2
layer3 = keras.layers.Conv2D(
    32,
    kernel_size=(3, 3),
    padding="same",
    activation="relu",
    kernel_regularizer=keras.regularizers.l2(0.06),
)
layer9 = keras.layers.Conv2D(
    32,
    kernel_size=(3, 3),
    padding="same",
    activation="relu",
    kernel_regularizer=keras.regularizers.l2(0.06),
)
layer4 = keras.layers.MaxPool2D(2)  # changed from 2
# layer31 = keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer=keras.regularizers.l2(0.06))
# layer91 = keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer=keras.regularizers.l2(0.06))
# layer41 = keras.layers.MaxPool2D(2) #changed from 2
layer5 = keras.layers.Flatten()
layer6 = keras.layers.Dense(
    300, activation="relu"
)  # ,kernel_regularizer=keras.regularizers.l2(0.05))
layer7 = keras.layers.Dense(128, activation="relu")
layer71 = keras.layers.Dense(64, activation="relu")
layer72 = keras.layers.Dense(32, activation="relu")
layer8 = keras.layers.Dense(5, activation="softmax")
layers = [
    layer1,
    layer2,
    layer3,
    layer9,
    layer4,
    layer5,
    layer6,
    layer7,
    layer71,
    layer72,
    layer8,
]

# Parameters
n_splits = 5  # Number of folds for cross-validation

# Initialize KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=302)

# Initialize lists to store training and validation results
train_acc_list = []
val_acc_list = []

for fold, (train_idx, val_idx) in enumerate(kf.split(feat_tr)):
    print(f"Training fold {fold + 1}/{n_splits}...")

    # Split the data into training and validation sets for this fold
    feat_tr_fold, feat_val_fold = feat_tr[train_idx], feat_tr[val_idx]
    targ_tr_fold, targ_val_fold = targ_tr_one_hot[train_idx], targ_tr_one_hot[val_idx]

    # Build a new model for each fold
    model = keras.Sequential(layers)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Fit the model
    history = model.fit(
        feat_tr_fold,
        targ_tr_fold,
        epochs=1000,
        validation_data=(feat_val_fold, targ_val_fold),
        batch_size=5,
        callbacks=[earlystopping, my_lr_scheduler],
    )

    # Evaluate the model on the validation set
    loss_val, acc_val = model.evaluate(feat_val_fold, targ_val_fold)
    loss_tr, acc_tr = model.evaluate(feat_tr_fold, targ_tr_fold)

    # Store the results
    train_acc_list.append(acc_tr)
    val_acc_list.append(acc_val)
    print(f"Fold {fold + 1} - Validation accuracy: {acc_val:.4f}")
    print(f"Fold {fold + 1} - Training accuracy: {acc_tr:.4f}")

# Calculate and print the average accuracy across all folds
avg_train_acc = np.mean(train_acc_list)
avg_val_acc = np.mean(val_acc_list)
print(f"\nAverage Training accuracy : {avg_train_acc:.4f}")
print(f"Average Validation accuracy : {avg_val_acc:.4f}")


# In[ ]:


model.save("CNN_"+save_name)


# In[16]:


np.save("/home/3155/"+save_name+"_loss_history", history.history["loss"])


# In[17]:


np.save("/home/3155/"+save_name+"_val_loss_history", history.history["val_loss"])
np.save("/home/3155/"+save_name+"_accuracy_history", history.history["accuracy"])
np.save("/home/3155/"+save_name+"_val_accuracy_history", history.history["val_accuracy"])


# In[ ]:


print("DONE TRAINING")
