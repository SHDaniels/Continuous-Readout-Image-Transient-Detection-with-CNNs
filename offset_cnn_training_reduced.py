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
from astropy.io import fits
import scipy.ndimage as snm
from tqdm.auto import tqdm
import os
from pylab import cm
import matplotlib as mpl
import sys
from sklearn.model_selection import KFold


save_name = sys.argv[1]





fpath_streak = "/lustre/lrspec/ZTF_data/training_feb26/offcenter_stamps_streaks/"


# for i, file in enumerate(os.listdir(fpath_streak)):
#     image_filename = os.fsdecode(file)
#     image_filename_full = fpath_streak + image_filename
#     print(image_filename)
#     streak = np.load(image_filename_full)
#     streak_arr = np.array(streak)
#     print(np.shape(streak))
#     # all_trans_stamps.append(trans_arr)
#     if i == 0:
#         all_streak_stamps = streak_arr
#     else:
#         all_streak_stamps = np.concatenate((all_streak_stamps, streak_arr), axis=0)

image_path = fpath_streak + "images/"
info_path = fpath_streak + "info/"

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
        # Add file index as first column
    file_idx = np.full((infos_arr.shape[0], 1), i)
    infos_arr = np.concatenate([file_idx, infos_arr], axis=1)
    
    print(np.shape(trans))
    print(np.shape(infos))
    # all_trans_stamps.append(trans_arr)
    if i == 0:
        all_streak_stamps = trans_arr
        all_streak_stamps_info = infos_arr
    else:
        all_streak_stamps = np.concatenate((all_streak_stamps, trans_arr), axis=0)
        all_streak_stamps_info = np.concatenate((all_streak_stamps_info, infos_arr), axis=0)


# In[4]:


np.shape(all_streak_stamps)


# In[5]:


fpath_trans = "/lustre/lrspec/ZTF_data/training_feb26/offcenter_stamps_pos/"

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
    
        # Add file index as first column
    file_idx = np.full((infos_arr.shape[0], 1), i)
    infos_arr = np.concatenate([file_idx, infos_arr], axis=1)
    
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


fpath_trans = "/lustre/lrspec/ZTF_data/training_feb26/offcenter_stamps_neg/"

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
        # Add file index as first column
    file_idx = np.full((infos_arr.shape[0], 1), i)
    infos_arr = np.concatenate([file_idx, infos_arr], axis=1)
    
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


fpath_bg = "/lustre/lrspec/ZTF_data/training_feb26/offcenter_stamps_background/"

# all_trans_stamps =[] #np.zeros((len(os.listdir(fpath_trans)), 10, 20, 20))

# for i, file in enumerate(os.listdir(fpath_bg)):
#     image_filename = os.fsdecode(file)
#     image_filename_full = fpath_bg + image_filename
#     print(image_filename)
#     bg = np.load(image_filename_full)
#     bg_arr = np.array(bg)
#     print(np.shape(bg))
#     # all_trans_stamps.append(trans_arr)
#     if i == 0:
#         all_bg_stamps = bg
#     else:
#         all_bg_stamps = np.concatenate((all_bg_stamps, bg_arr), axis=0)

image_path = fpath_bg + "images/"
info_path = fpath_bg + "info/"

for i, file in enumerate(os.listdir(image_path)):
    image_filename = os.fsdecode(file)
    image_filename_full = image_path + image_filename
    print(image_filename)
    info_filename = image_filename[:50]  + "info_"+image_filename[50:]
    info_filename_full = info_path + info_filename
    trans = np.load(image_filename_full)
    trans_arr = np.array(trans)
    infos = np.load(info_filename_full)
    infos_arr = np.array(infos)
        # Add file index as first column
    file_idx = np.full((infos_arr.shape[0], 1), i)
    infos_arr = np.concatenate([file_idx, infos_arr], axis=1)
    

    # all_trans_stamps.append(trans_arr)
    if i == 0:
        all_bg_stamps = trans_arr
        all_bg_stamps_info = infos_arr
    else:
        all_bg_stamps = np.concatenate((all_bg_stamps, trans_arr), axis=0)
        all_bg_stamps_info = np.concatenate((all_bg_stamps_info, infos_arr), axis=0)




def split_stamps(stamps, info, train_size, random_state, return_test=False, numinid=5, numthesame=5, return_mask=False):
    transient_ids = np.unique(info[:, :numinid], axis=0)
    if train_size==.75:
            # Split the transient IDs
        train_ids, test_ids = train_test_split(
            transient_ids, 
            train_size=train_size,
            random_state=random_state
        )
    else:
        # Split the transient IDs
        train_ids, test_ids = train_test_split(
            transient_ids, 
            train_size=int(train_size/numthesame),
            random_state=random_state
        )

    # Create a mask for training data
    train_mask = np.isin(
        info[:, :numinid].view([('', info.dtype)] * numinid),
        train_ids.view([('', train_ids.dtype)] * numinid)
    )
    if return_mask:
        return train_mask.flatten()
    # Split your stamps
    stamps_use = stamps[train_mask.flatten()]
    bness_use = info[:, 1][train_mask.flatten()]
    lens_use = info[:, 2][train_mask.flatten()]
    if return_test:
        return stamps_use, bness_use, lens_use, stamps[~train_mask.flatten()], info[:, 0][~train_mask.flatten()], info[:, 1][~train_mask.flatten()] #train, test
    else:
        return stamps_use, bness_use, lens_use, info[train_mask.flatten()]
    

pos_use, bness_pos_use, lens_pos_use, info_use_pos = split_stamps(all_pos_stamps, all_pos_stamps_info, 57400/6, random_state=301, return_test=False)

neg_use, bness_neg_use, lens_neg_use,info_use_neg = split_stamps(all_neg_stamps, all_neg_stamps_info, 57400/6, random_state=302, return_test=False)
   
streaks_use, ignore, ignore2, info_use_streak = split_stamps(all_streak_stamps, all_streak_stamps_info, 57400/6, random_state=303, return_test=False)

bg_use, ignore, ignore2, info_use_bg = split_stamps(all_bg_stamps, all_bg_stamps_info, 57400/3, random_state=304, return_test=False)
  

    

pos_train, bness_train_pos, lens_train_pos, pos_test, bness_test_pos,lens_test_pos = split_stamps(pos_use, info_use_pos, .75, random_state=301, return_test=True)
neg_train, bness_train_neg, lens_train_neg, neg_test, bness_test_neg,lens_test_neg = split_stamps(neg_use, info_use_neg, .75, random_state=302, return_test=True)
streak_train, ignore, ignore, streak_test, ignore,ignore = split_stamps(streaks_use, info_use_streak, .75, random_state=303, return_test=True)
bg_train, bness_train, lens_train, bg_test, bness_test,lens_test = split_stamps(bg_use, info_use_bg, .75, random_state=304, return_test=True)

cr_train, cr_test = train_test_split(all_cr_stamps, random_state=305)


# Combine stamps and labels
all_train_stamps = np.concatenate([bg_train, pos_train, neg_train, cr_train, streak_train], axis=0)
all_test_stamps = np.concatenate([bg_test, pos_test, neg_test, cr_test, streak_test], axis=0)

# Create labels (e.g. 1=streak, 0=bg)
streak_train_labels = np.ones(len(bg_train), dtype=int)
bg_train_labels = np.zeros(len(bg_train), dtype=int)
all_train_labels = np.concatenate([np.zeros(len(bg_train), dtype=int), 
                                   np.full(len(pos_train), 1),
                                 np.full(len(neg_train), 2),
                                 np.full(len(cr_train), 3),
                                 np.full(len(streak_train), 4)])


all_test_labels = np.concatenate([np.zeros(len(bg_test), dtype=int), 
                                   np.full(len(pos_test), 1),
                                 np.full(len(neg_test), 2),
                                 np.full(len(cr_test), 3),
                                 np.full(len(streak_test), 4)])

# # Also combine info if you need it
# all_train_info = np.concatenate([streak_train_info, bg_train_info], axis=0)
# all_test_info = np.concatenate([streak_test_info, bg_test_info], axis=0)

# Process bness and lens
bness_train = np.concatenate(
    (
        np.full(len(bg_train), 0.),
        bness_train_pos,
        bness_train_neg,
        np.full(len(cr_train), 0.),
        np.full(len(streak_train), 0.),
    )
)

lens_train = np.concatenate(
    (
        np.full(len(bg_train), 0.),
        lens_train_pos,
        lens_train_neg,
        np.full(len(cr_train), 0.),
        np.full(len(streak_train), 0.),
    )
)

# Process bness and lens
bness_test = np.concatenate(
    (
        np.full(len(bg_test), 0.),
        bness_test_pos,
        bness_test_neg,
        np.full(len(cr_test), 0.),
        np.full(len(streak_test), 0.),
    )
)

lens_test = np.concatenate(
    (
        np.full(len(bg_test), 0.),
        lens_test_pos,
        lens_test_neg,
        np.full(len(cr_test), 0.),
        np.full(len(streak_test), 0.),
    )
)

# Shuffle using a shared permutation index
train_perm = np.random.default_rng(seed=301).permutation(len(all_train_stamps))
test_perm = np.random.default_rng(seed=301).permutation(len(all_test_stamps))

feat_tr = all_train_stamps[train_perm]
targ_tr = all_train_labels[train_perm]
bness_tr   = bness_train[train_perm]
lens_tr = lens_train[train_perm]

feat_te = all_test_stamps[test_perm]
targ_te = all_test_labels[test_perm]
bness_te  = bness_test[test_perm]
lens_te  = lens_test[test_perm]

print(f"Train: {len(all_train_stamps)} stamps, {all_train_labels.sum()} streaks, {(all_train_labels==0).sum()} bg")
print(f"Test:  {len(all_test_stamps)} stamps, {all_test_labels.sum()} streaks, {(all_test_labels==0).sum()} bg")



# Reshape features for the CNN
feat_tr = feat_tr.reshape(feat_tr.shape[0], feat_tr.shape[1], feat_tr.shape[2], 1)
feat_te = feat_te.reshape(feat_te.shape[0], feat_te.shape[1], feat_te.shape[2], 1)


# # Split targets and additional variables
# bness_tr, bness_te = train_test_split(bness, random_state=302)
# lens_tr, lens_te = train_test_split(lens, random_state=302)
# targ_tr, targ_te = train_test_split(targ, random_state=302)

# One-hot encode the target variable
targ_tr_one_hot = to_categorical(targ_tr, num_classes=5)
targ_te_one_hot = to_categorical(targ_te, num_classes=5)








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
