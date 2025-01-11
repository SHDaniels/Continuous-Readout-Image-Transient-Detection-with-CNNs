#imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as snm
from tqdm.auto import tqdm
import os
from pylab import cm
import matplotlib as mpl
import get_postage_stamps as gpsf
import gc #garbage collector

def check_if_true(loc, offset, file_info):
    '''
    this function will take a row and column
    (of the center of mass of the found objects)
    and return "True" if they are in the list of implanted objects
    or "False" if not
    NB include offset as the width of the postage stamp if that offset has 
    not already been accounted for. if it has, put zero there.
    loc: (row, column)
    file_info: the info from the file where the info about the implanted transient lives 
    (just np.load it in as an array)
    '''
    row, col = loc
    for i in range(np.shape(file_info)[0]):
        rowtocheck = file_info[i,1]
        coltocheck = file_info[i,0]
        col_diff = np.abs(col + offset - coltocheck)
        row_diff = np.abs(row + offset - rowtocheck)
        if col_diff<3 and row_diff<3: #the largest i have seen is .6 but i want to be a little generous with it
            #print(col_diff)
            #print(row_diff)
            return True
    return False

def precision(num_true_retrieved, num_all_retrieved):
    '''
    precision: relevant retrieved instances over all retrieved instances
    both inputs are ints or floats
    returns float
    '''
    return num_true_retrieved/num_all_retrieved

def recall(num_true_retrieved, num_all_true):
    '''
    recall: relevant retrieved instances over all relevant instances
    both inputs are ints or floats
    returns float
    '''
    return num_true_retrieved/num_all_true
    
    
def get_num_true_retrieved(loc_list, offset, file_info):
    '''
    this function gets the number of implanted transients that were found by the CNN
    loc_list: list of transient locations the nn found
    offset: (half) width of the postage stamp *if* the offset has not already been accounted for
    (because the probability output from the CNN is smaller by this much than the original image)
    file_info: the info from the file where the info about the implanted transient lives 
    (just np.load it in as an array) 
    returns an int (number of transients)
    '''
    counter_tp = 0
    counter_fp = 0
    for i in range(len(loc_list)):
        if check_if_true(loc_list[i], offset, file_info):
            counter_tp += 1
        else:
            counter_fp += 1
    return counter_tp, counter_fp

#make an array in that function

def get_tpfptnfn(detections, rowthreshmax, rowthreshmin, colthreshmax, colthreshmin, file_info):
    '''
    This functions gets the number of true positive, false positive, true negative, and false negative
    detections in an array of detections given file_info with the true info.
    
    detections: boolean array of detections, for instance detectionssquare
    rowthreshmax: this is the maximum vertical distance that a positive detection can be from a true implantation
    in order for it to count as a true positive
    rowthreshmin: this is the maximum vertical distance that a negative detection can be from a true implantation
    in order for it to count as a true negative
    colthreshmax: this is the maximum horizontal distance that a positive detection can be from a true implantation
    in order for it to count as a true positive
    colthreshmin: this is the maximum horizontal distance that a negative detection can be from a true implantation
    in order for it to count as a true negative    
    this is what i have been using so far:
    colthreshmax = 4
    colthreshmin = 1.5
    rowthreshmin = 3
    rowthreshmax = 15 # 4 stdvs
    i can hard code it at some point.
    file_info: the info from the file where the info about the implanted transient lives 
    (just np.load it in as an array) 
    
    returns: 
    sums: the totals of the following array, so total tp, total fp, total tn, total fn
    array: boolean array for every pixel in detections where one of the following four will be true:
    true positive, false positive, true negative, and false negative (in that order)
    '''
    wid = 10 #should make this an input
    #true positive, false positive, true negative, false negative
    arr = np.zeros((detections.shape[0]*detections.shape[1], 4), bool)
    somethinghaschanged = False
    for j in tqdm(range(detections.shape[1])): #for every column
        for i in range(detections.shape[0]): #for every row
        #i do not know how to loop over a 2d array...welll...i guess i do but it is uglier 
            #print("i = "+str(i)+" j = "+str(j))
            if detections[i,j] == True: #check on order of indices
                for el in file_info:   
                    if np.abs(el[1] - wid - i) < rowthreshmax and np.abs(el[0] - wid - j) < colthreshmax: #true and close
                        #el[0] is the column nummber
                        #el[1] is the row number
                        arr[j * detections.shape[0] + i, 0] = True #true positive
                        #print("true positive! i = "+str(i)+"j = "+str(j))
                        somethinghaschanged = True
                        break
                if somethinghaschanged==False:
                    arr[j * detections.shape[0]+i, 1] = True #false positive
            else: #detection is false
                for el in file_info:
                    if np.abs(el[1] - wid -i) < rowthreshmin and np.abs(el[0] - wid -j) < colthreshmin: #close but marked false
                        arr[j * detections.shape[0]+i, 3] = True #false negative
                        somethinghaschanged = True
                        break
                if somethinghaschanged==False:
                    arr[j * detections.shape[0]+i, 2] = True #true negative
            somethinghaschanged = False
        somethinghaschanged = False
        
    sums = np.sum(arr,axis=0)
    return sums, arr




def get_num_all_true(file_info):
    '''
    get the number of total implanted transients
    file_info: the info from the file where the info about the implanted transient lives 
    (just np.load it in as an array) 
    returns an int (number of transients)
    '''
    return(len(file_info))

def get_num_all_retrieved(loc_list):
    '''
    get the number of transients that the NN found
    loc_list: list of transient locations the nn found
    returns an int (number of transients)
    '''
    return len(loc_list)

def precision_thresh(thresh, pimg, file_info):
    '''
    precision: relevant retrieved instances over all retrieved instances
    this determines the precision at a certain threshold for counting 
    a transient probability as "transient"
    thresh: the probability threshold for counting as a transient
    pimg: probability image (note that it takes the transposed pimg as input
    since that is commonly output by other processes)
    file_info: the info from the file where the info about the implanted transient lives 
    (just np.load it in as an array)     
    returns precision (float)
    '''
    threshed = pimg.T>thresh #it takes the transposed pimg
    labs, nlabs = snm.label(threshed)
    loc_list = snm.center_of_mass(threshed,labs,range(1,nlabs+1))
    num_all_retrieved = get_num_all_retrieved(loc_list)
    num_true_retrieved = get_num_true_retrieved(loc_list, 10, file_info)
    preci = precision(num_true_retrieved, num_all_retrieved)
    return preci   

def recall_thresh(thresh, pimg, file_info):
    '''
    recall: relevant retrieved instances over all relevant instances
    this determines the recall at a certain threshold for counting 
    a transient probability as "transient"
    thresh: the probability threshold for counting as a transient
    pimg: probability image (note that it takes the transposed pimg as input
    since that is commonly output by other processes)
    file_info: the info from the file where the info about the implanted transient lives 
    (just np.load it in as an array)     
    returns recall (float)
    '''

    threshed = pimg.T>thresh #it takes the transposed pimg
    labs, nlabs = snm.label(threshed)
    loc_list = snm.center_of_mass(threshed,labs,range(1,nlabs+1))
    num_true_retrieved = get_num_true_retrieved(loc_list, 10, file_info)
    num_all_true = get_num_all_true(file_info)
    rec = recall(num_true_retrieved, num_all_true)
    return rec   

def get_objsizes(labs, nlabs):
    '''
    this gets the size of labeled objects
    returns an array corresponding to 
    [size of object 1, size of object 2, size of...]
    NB that object zero is excluded because it is almost always background
    labs: labeled np array
    nlabs: float number of labels
    returns: array of int sizes
    '''
    objsizes = np.array([0 for a in range(nlabs)])
    for i in range(nlabs):
        howmany = ((i+1) == labs).sum()
        objsizes[i] = howmany
    return objsizes

def precision_thresh_size_criterion(thresh, pimg, file_info, size_criterion):
    '''
    precision: relevant retrieved instances over all retrieved instances
    this determines the precision at a certain threshold for counting 
    a transient probability as "transient" 
    thresh: the probability threshold for counting as a transient
    pimg: probability image (note that it takes the transposed pimg as input
    since that is commonly output by other processes)
    file_info: the info from the file where the info about the implanted transient lives 
    (just np.load it in as an array)  
    size_criterion: how large a transient has to be (pixels) in order for it to count as a transient
    returns precision (float)
    '''
    threshed = pimg.T>thresh #it takes the transposed pimg
    labs, nlabs = snm.label(threshed)
    loc_list = snm.center_of_mass(threshed,labs,range(1,nlabs+1))
    objsizes = get_objsizes(labs, nlabs)
    mask = objsizes > size_criterion
    #print(mask)
    new_loc_list = [] #i am aware that there is a much cleaner way to do this but i do not know how
    for i in range(len(mask)):
        if mask[i]:
            new_loc_list.append(loc_list[i])
    
    #print(new_loc_list)
    num_all_retrieved = get_num_all_retrieved(new_loc_list)
    num_true_retrieved = get_num_true_retrieved(new_loc_list, 10, file_info)
    preci = precision(num_true_retrieved, num_all_retrieved)
    return preci   

def recall_thresh_size_criterion(thresh, pimg, file_info, size_criterion):
    '''
    recall: relevant retrieved instances over all relevant instances
    this determines the recall at a certain threshold for counting 
    a transient probability as "transient"
    thresh: the probability threshold for counting as a transient
    pimg: probability image (note that it takes the transposed pimg as input
    since that is commonly output by other processes)
    file_info: the info from the file where the info about the implanted transient lives 
    (just np.load it in as an array)     
    returns recall (float)
    size_criterion: how large a transient has to be (pixels) in order for it to count as a transient
    returns precision (float)
    '''

    threshed = pimg.T>thresh #it takes the transposed pimg
    labs, nlabs = snm.label(threshed)
    loc_list = snm.center_of_mass(threshed,labs,range(1,nlabs+1))
    objsizes = get_objsizes(labs, nlabs)
    mask = objsizes > size_criterion
    #print(mask)
    new_loc_list = [] #i am aware that there is a much cleaner way to do this but i do not know how
    for i in range(len(mask)):
        if mask[i]:
            new_loc_list.append(loc_list[i])
    
    #print(new_loc_list)

    num_true_retrieved = get_num_true_retrieved(new_loc_list, 10, file_info)
    num_all_true = get_num_all_true(file_info)
    rec = recall(num_true_retrieved, num_all_true)
    return rec   

def zoominthere(loc, image, pimg, offset):
    '''
    sometimes i see a transient location come up and i'm like
    what is that. can we zoom in there
    loc: (row, column)
    image: array (streak image)
    pimg: array (probability image)
    offset: difference between rows/cols of original and pimg (usually it is wid (10))
    returns: nothing but shows plot of zoomed in
    '''
    wid = offset
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    row,col=loc
    ax[0].imshow(np.sqrt(image))
    ax[0].set_xlim(col+wid-20,col+wid+20)
    ax[0].set_ylim(row-20+wid,row+20+wid)
    #plt.colorbar()
    ax[1].imshow(pimg.T)
    ax[1].set_xlim(col-20,col+20)
    ax[1].set_ylim(row-20,row+20)
    fig.suptitle("zoom in on pimg")
    
def make_found_transient_list(image, model, abst_thr, mult_thr, wid, bg_thresh, path):
    '''
    This function will take an image and a cnn model and return an array
    of the transients found in that image.
    image: np array
    model: cnn model
    abst_thr: absolute probability threshold to consider a transient probability as a transient
    mult_thr: comparative probability threshold to consider a transient probability as a transient
    (aka the trans prob has to be x times larger than (streak prob + backg prob)
    wid: desired postage stamp width
    bg_thresh: thresh to consider the average background on a column to be "background",
    something like .1, used as criterion for determining cosmic rays
    path: path to save the array
    
    returns (and saves): array of length (number of transient detections),
    with elements of (detection row, detection column), transient detection size, 
    realness (aka not-cosmic-ray)
    so a list of tuples, then an int, then 0 or 1
    0 means it was flagged as a probable cosmic ray
    '''


    nrow, ncol = image.shape  
    print("getting postage stamps")
    stamps = gpsf.get_postage_stamps_full(image, 10)
    stampbatches = np.array_split(stamps, 
                          stamps.shape[0] // 20000) #(image.shape[0] - 2 * wid) ) 
    #I WANNA CHANGE THIS TO SPEED HIM UP
    #so to be faster there should be less batches
    #so i want the second argument to be smaller
    #so i want the denomitor of the second argument to be larger
    preds = []
    print("getting predictions:")
    for stamps in tqdm(stampbatches): #getting all the predictions
        pred_l = model.predict_on_batch(stamps);
        preds.append(pred_l)
    predsarray = np.concatenate(preds) 
    #i would like to save the predsarrays i think. for reproducibility
    np.save(path+"_predsarray", predsarray)
    del stamps
    del stampbatches
    del preds
    gc.collect()
    pimg = predsarray[:,1].reshape(ncol-2*wid, nrow-2*wid) 
    
    pimg_bg = predsarray[:,0].reshape(ncol-2*wid, nrow-2*wid) 
    pimg_streak = predsarray[:,2].reshape(ncol-2*wid, nrow-2*wid) 
    #plt.imshow(pimg_bg.T)
    #plt.colorbar()
    del predsarray
    gc.collect() #trying to save memory here...
    # plt.imshow(pimg.T)
    # plt.show()
    # plt.close()
    thrdet = pimg > abst_thr
    compdet = pimg > (mult_thr * pimg_bg + mult_thr * pimg_streak)
    threshed0 = np.where(compdet==thrdet, compdet, False)
    threshed = threshed0.T #it takes the transposed pimg
    labs, nlabs = snm.label(threshed)
    loc_list = snm.center_of_mass(threshed,labs,range(1,nlabs+1))
    
    objsizes=get_objsizes(labs, nlabs)
    # bg_thresh=.1
    cosmicray=np.full(len(loc_list),1)
    for k, loc in enumerate(loc_list):
        (row, col) = loc
        # print("loc = row,col= "+str(loc))
        # print(row)
        # print(col)
        colavg = np.average(pimg_bg.T[:,int(col)])
#        print("colavg: "+str(colavg) )               
        if colavg>bg_thresh: #column is mostly background - can adjust this cutoff
            cosmicray[k] = 0

    loc_size_ray=np.array((loc_list, objsizes, cosmicray)).T
    del pimg
    del pimg_bg
    del pimg_streak
    del thrdet
    del compdet
    del threshed0
    del loc_list
    del objsizes
    del cosmicray
    del labs
    del threshed
    gc.collect()
    np.save(path, loc_size_ray)
    return loc_size_ray
    
    
def transient_list_newthr(predsarray):
#(image, model, thresh, wid, bg_thresh, path):
    '''
    THIS DOESNT DO ANYTHING RN
    This function will take an image and a cnn model and return an array
    of the transients found in that image.
    image: np array
    model: cnn model
    thresh: probability threshold to consider a transient probability as a transient
    wid: desired postage stamp width
    bg_thresh: thresh to consider the average background on a column to be "background",
    something like .1, used as criterion for determining cosmic rays
    path: path to save the array
    
    returns (and saves): array of length (number of transient detections),
    with elements of (detection row, detection column), transient detection size, 
    realness (aka not-cosmic-ray)
    so a list of tuples, then an int, then 0 or 1
    1 means it was flagged as a probable cosmic ray
    '''
    thr=thresh
    nrow, ncol = image.shape  

    threshed = pimg.T>thresh #it takes the transposed pimg
    labs, nlabs = snm.label(threshed)
    loc_list = snm.center_of_mass(threshed,labs,range(1,nlabs+1))
    
    objsizes = get_objsizes(labs, nlabs)
    # bg_thresh=.1
    cosmicray = np.full(len(loc_list),1)
    for k, loc in enumerate(loc_list):
        (row, col) = loc
        # print("loc = row,col= "+str(loc))
        # print(row)
        # print(col)
        colavg = np.average(pimg_bg.T[:,int(col)])
#        print("colavg: "+str(colavg) )               
        if colavg>bg_thresh: #column is mostly background - can adjust this cutoff
            cosmicray[k] = 0

    loc_size_ray=np.array((loc_list, objsizes, cosmicray)).T
    del pimg
    del pimg_bg
    del loc_list
    del objsizes
    del cosmicray
    del labs
    del threshed
    gc.collect()
    np.save(path, loc_size_ray)
    return loc_size_ray
    #return locs_and_objsizes