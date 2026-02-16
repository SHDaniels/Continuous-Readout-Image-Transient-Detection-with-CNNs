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
    info_filename = image_filename[:-4] + "_info.npy"
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


#STREAKS       
# Get unique transient identifiers (first 5 columns now)
transient_ids = np.unique(all_streak_stamps_info[:, :5], axis=0)

# Split the transient IDs
train_ids, test_ids = train_test_split(
    transient_ids, 
    train_size=int(57400/6/5),
    random_state=301
)

# Create a mask for training data
train_mask = np.isin(
    all_streak_stamps_info[:, :5].view([('', all_streak_stamps_info.dtype)] * 5),
    train_ids.view([('', train_ids.dtype)] * 5)
)

# Split your stamps
streak_use = all_streak_stamps[train_mask.flatten()]
# streak_test = all_streak_stamps[~train_mask.flatten()]

streak_info_use = all_streak_stamps_info[train_mask.flatten()]
# streak_test_info = all_streak_stamps_info[~train_mask.flatten()]

#BACKGROUND     
# Get unique transient identifiers (first 5 columns now)
transient_ids = np.unique(all_bg_stamps_info[:, :5], axis=0)

# Split the transient IDs
train_ids, test_ids = train_test_split(
    transient_ids, 
    train_size=int(57400/3/5),
    random_state=302
)

# Create a mask for training data
train_mask = np.isin(
    all_bg_stamps_info[:, :5].view([('', all_bg_stamps_info.dtype)] * 5),
    train_ids.view([('', train_ids.dtype)] * 5)
)

# Split your stamps
bg_use = all_bg_stamps[train_mask.flatten()]
# streak_test = all_streak_stamps[~train_mask.flatten()]

bg_info_use = all_bg_stamps_info[train_mask.flatten()]
# streak_test_info = all_streak_stamps_info[~train_mask.flatten()]


#POS     
# Get unique transient identifiers (first 5 columns now)
transient_ids = np.unique(all_pos_stamps_info[:, :5], axis=0)

# Split the transient IDs
train_ids, test_ids = train_test_split(
    transient_ids, 
    train_size=int(57400/6/5),
    random_state=303
)

# Create a mask for training data
train_mask = np.isin(
    all_pos_stamps_info[:, :5].view([('', all_pos_stamps_info.dtype)] * 5),
    train_ids.view([('', train_ids.dtype)] * 5)
)

# Split your stamps
pos_use = all_pos_stamps[train_mask.flatten()]
# streak_test = all_streak_stamps[~train_mask.flatten()]

# pos_train_use = all_pos_stamps_info[train_mask.flatten()]
# streak_test_info = all_streak_stamps_info[~train_mask.flatten()]

bness_pos_use = all_pos_stamps_info[:, 0][train_mask.flatten()]

lens_pos_use = all_pos_stamps_info[:, 1][train_mask.flatten()]
# bg_use, bg_extra = train_test_split(all_bg_stamps, train_size=int(57400/3), random_state=301) #train size = ones we're using

# pos_use, pos_extra, bness_pos_use, bness_pos_extra, lens_pos_use, lens_pos_extra = train_test_split(all_pos_stamps, 
#                                                                                                    all_pos_stamps_info[:, 0], 
#                                                                                                    all_pos_stamps_info[:, 1],  
#                                                                                                    train_size=int(57400/6),
#                                                                                                    random_state=303)
# neg_use, neg_extra, bness_neg_use, bness_neg_extra, lens_neg_use, lens_neg_extra = train_test_split(all_neg_stamps, 
#                                                                                                    all_neg_stamps_info[:, 0], 
#                                                                                                    all_neg_stamps_info[:, 1],  
#                                                                                                    train_size=int(57400/6),
#                                                                                                    random_state=304)
# streaks_use, streaks_extra = train_test_split(all_streak_stamps,
#                                               train_size=int(57400/6),
#                                               random_state=305) #was 304 when everything was divided by 2 instead of 6



def split_stamps(stamps, info, train_size, random_state, return_test=False):
    transient_ids = np.unique(info[:, :5], axis=0)

    # Split the transient IDs
    train_ids, test_ids = train_test_split(
        transient_ids, 
        train_size=int(train_size/5),
        random_state=random_state
    )

    # Create a mask for training data
    train_mask = np.isin(
        info[:, :5].view([('', info.dtype)] * 5),
        train_ids.view([('', train_ids.dtype)] * 5)
    )
    
    # Split your stamps
    stamps_use = stamps[train_mask.flatten()]
    bness_use = info[:, 0][train_mask.flatten()]
    lens_use = info[:, 1][train_mask.flatten()]
    if return_test:
        return stamps_use, bness_use, lens_use, stamps[~train_mask.flatten()], info[:, 0][~train_mask.flatten()], info[:, 1][~train_mask.flatten()] #train, test
    else:
        return stamps_use, bness_use, lens_use

    
neg_use, bness_neg_use, lens_neg_use = split_stamps(all_neg_stamps, all_neg_stamps_info, 57400/6, random_state=304, return_test=False)
    




# Process bness and lens
bness = np.concatenate(
    (
        np.full(len(bg_use), 0),
        bness_pos_use,
        bness_neg_use,
        np.full(len(all_cr_stamps), 0),
        np.full(len(streaks_use), 0),
    )
)

lens = np.concatenate(
    (
        np.full(len(bg_use), 0),
        lens_pos_use,
        lens_neg_use,
        np.full(len(all_cr_stamps), 0),
        np.full(len(streaks_use), 0),
    )
)

targ = np.concatenate(
    (
        np.full(len(bg_use), 0),
        np.full(len(pos_use), 1),
        np.full(len(neg_use), 2),
        np.full(len(all_cr_stamps), 3),
        np.full(len(streaks_use), 4),
    )
)


# Combine all features
feat = np.concatenate(
    (bg_use, pos_use, neg_use, all_cr_stamps, streaks_use),
    axis=0,
)


# Split features into training and testing sets once
feat_tr, feat_te, bness_tr, bness_te, lens_tr, lens_te, targ_tr, targ_te = train_test_split(feat, bness, lens, targ, random_state=302)



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
