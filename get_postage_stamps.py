import numpy as np
from tqdm.auto import tqdm

def get_postage_stamps_full(image, wid):
    '''
    this function takes an input image 
    image_filename_full: full name of image file (images are already normalized)

    wid: half the size of stamp that the model is trained on (should be 10)
    output: array of center values of detected transients'''
    

    nrow, ncol = image.shape
    stamps = np.zeros(((nrow - 2 * wid) * (ncol - 2 * wid), 
                      2 * wid, 2 * wid))

    kk = 0
    print("getting stamps now")
    for ii in tqdm(range(wid, ncol-wid)):
        for jj in range(wid, nrow-wid):
            stamps[kk] = image[jj-wid:jj+wid,  ii-wid:ii+wid]
            kk += 1


    return stamps
    
def getpredsarray(stamps, model):
    '''
    This function takes stamps and outputs a prediction array for each stamp
    stamps: probably from get_postage_stamps_full
    model: neural network
    output: array of predictions (not image shaped, 1x3 or something)
    '''
    stampbatches = np.array_split(stamps, 
                          stamps.shape[0] // (20000))
    preds = []
    print("getting predictions")
    for stamps in tqdm(stampbatches): #getting all the predictions
        pred_l = model.predict_on_batch(stamps);
        preds.append(pred_l)
    predsarray = np.concatenate(preds) 
    return predsarray
    
def getpimg(num, predsarray, wid, nrowcol):
    '''
    This function takes a prediction array and outputs a probability image
    shaped like the original image (minus wid on all sides)
    num: which prediction output you want the probability image for:
        0 - background
        1 - transient
        2 - streak
    predsarray: array of nn predictions (not image shaped, 1x3 or something)
    wid: half the size of stamp that the model is trained on (should be 10)
    nrowncol: tuple number of rows and columns in original image (output of "image.shape")
    outputs: reshaped array of predsarray to match original image
    NOTE: previous versions returned pimg.T and this one does not
    '''
    nrow, ncol = nrowcol
    pimg = predsarray[:,num].reshape(ncol-2*wid, nrow-2*wid)
    return pimg.T