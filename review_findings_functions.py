import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import sobel

def getdf(direcname):
    '''
    deprecated
    '''
    fpath_extractions = "/lustre/lrspec/ZTF_data/science_extractions/"+direcname+"/"
    dfname = fpath_extractions+"df_detections_"+str(direcname)
    print(dfname)
    df = pd.read_csv(dfname)
    return df

def df_shorten(df, size=None, bgmax=None, brlim=None, num_desired=None):
    dfleninit = len(df)
    if size!=None:
        df = df[df["detection_size"]>size]
    if bgmax!=None:
        df = df[df["cosmic_ray_flag"]<bgmax]
    if brlim!=None:
        df = df[df["max brightness"]>brlim]
    if num_desired!=None:
        print("didn't set up num desired")
    print("Dataframe shortened from "+str(dfleninit)+" to "+str(len(df))+".")
    return df


def inspect_df(df, direcname):
    wid = 10
    figcols = 5
    figrows = 1+len(df)//figcols
    fig, ax = plt.subplots(figrows, figcols, figsize=(figcols*1.8, figrows*1.8))
    for i in range(figrows):
        for j in range(figcols):
            index = figcols*i+j
            if index<len(df):
                fname = df.iloc[index]["filename"]+"fits.fz"
                row = int(df.iloc[index]["row_from_nn"])
                col = int(df.iloc[index]["col_from_nn"])
                fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"_minus_extras/"
                path = fpath+fname
                hdul=fits.open(path)
                file1_array = hdul[1].data
                norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array))
                # ax[i, j].imshow(norm[col+wid-10:col+wid+10, row-10+wid:row+10+wid])
                ax[i, j].imshow(np.sqrt(norm[row-10+wid:row+10+wid, col+wid-10:col+wid+10]))
                ax[i, j].set_axis_off()
                # ax[i, j].set_title(fname[26:-8])
                ax[i, j].set_title(df.iloc[index]["Unnamed: 0"])
    plt.tight_layout()
    plt.show()
                # ax[i, j].set_ylim(col+wid-10,col+wid+10)
                # ax[i, j].set_xlim(row-10+wid,row+10+wid)
                #print(path)
                
                
def get_image_background(image):
    '''
    this function gets the background value of an array of various brightnesses
    and iteratively clips the maximum at a 3 standard deviation level to get the background value
    input: array of any dimensions
    return: background value (float)
    '''
    
    change = 100
    newnorm = image
    while np.abs(change) > .0001:
        stdv = np.std(newnorm)
        norm_mean = np.mean(newnorm) 
        newnorm = np.clip(newnorm, 0, norm_mean+3*stdv) #check - is this the right cutoff
        newnorm_mean = np.mean(newnorm)
        change = newnorm_mean-norm_mean
        background = newnorm_mean
    return background


# def get_rfft_sq(index, direcname, df):
#     fname = df.iloc[index]["filename"]+"fits.fz"
#     row = int(df.iloc[index]["row_from_nn"])+10
#     col = int(df.iloc[index]["col_from_nn"])+10
#     fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"_minus_extras/"
#     path = fpath+fname
#     hdul=fits.open(path)
#     file1_array = hdul[1].data
#     image = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array)) #norm
    
    
#     background = get_image_background(image)
    
#         #subtracting background
#     imagenobackground = image - np.full(np.shape(image),background)
#     averages = np.average(imagenobackground, axis=0)
#     mid = col
#     column = col
#     indices_to_check = [mid-4, mid-3, mid-2, mid-1, mid, mid+1, mid+2, mid+3] 
#     column_indices = []
#     for index in indices_to_check:
#         if averages[index]>.01:
#             column_indices.append(index)
#     #print(column_indices)
#     streaks = np.zeros(( len(column_indices), len(image[:,column])))
#     #print(streaks.shape)
#     for i, index in enumerate(column_indices):
#         streaks[i] = image[:,index]
#     streakavg = np.average(streaks, axis=0)
#     return np.square(np.real(np.fft.rfft(streakavg)[1:]))



    
    
# def plot_time_series_from_df_3(df, index, direcname):
#     '''


#     '''
#     # index=2
#     # df=df_0820_short
#     # direcname = "0820"
#     wid=10
#     fname = df.iloc[index]["filename"]+"fits.fz"
#     row = int(df.iloc[index]["row_from_nn"])+10
#     col = int(df.iloc[index]["col_from_nn"])+10
#     fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"_minus_extras/"
#     path = fpath+fname
#     hdul=fits.open(path)
#     file1_array = hdul[1].data
#     image = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array)) #norm
#     plt.rcParams["figure.figsize"] = (7, 7)
#     plt.rcParams["image.interpolation"] = "nearest"
#     plt.rcParams["image.cmap"] = "gist_gray"
    
#     column = int(col)
#     gs_kw = dict(width_ratios=[3, 2, 1], height_ratios=[1,  2])
#     fig, axd = plt.subplot_mosaic([['upper left', 'upper mid', 'upper right'],
#                                ['lower left', 'lower mid','lower right']],
#                               gridspec_kw=gs_kw, figsize=(7.5, 3.5),
#                               layout="constrained")
#     #note: the scatter plots are shifted in the positive x direction by wid, 
#     #so that they line up with the images
    
#     background = get_image_background(image)

#         #subtracting background
#     imagenobackground = image - np.full(np.shape(image),background)
#     averages = np.average(imagenobackground, axis=0)
#     mid = int(col)
#     indices_to_check = [mid-4, mid-3, mid-2, mid-1, mid, mid+1, mid+2, mid+3] 
#     column_indices = []
#     for ind in indices_to_check:
#         if averages[ind]>.01:
#             column_indices.append(ind)
    
#     streaks = np.zeros(( len(column_indices), len(image[:,column])))
#     print(column_indices)
#     for i, inde in enumerate(column_indices):
#         streaks[i] = image[:,inde]
#     streakavg = np.average(streaks, axis=0)
    
    
#     axd['lower left'].scatter(np.arange(len(streakavg)), streakavg,
#                 s=np.full(len(image[:,column]),6), color="black",
#                 marker="|")
#             # s=np.full(len(image[:,column])-2*wid,3), norm=mpl.colors.Normalize(vmin=0, vmax=6),
#             # marker="|", c=cimg.T[:,column-10], cmap="prism")
#     axd['upper left'].imshow(np.sqrt(image).T,aspect=80,cmap="gist_gray")
#     axd['upper left'].set_ylim(column-4,column+4)
#     axd['upper right'].imshow(np.sqrt(image).T,aspect=1.4,cmap="gist_gray")
#     axd['upper mid'].imshow(np.sqrt(image).T,aspect=1.4,cmap="gist_gray")
#     axd['upper right'].set_xlim(row-9,row+9)
#     axd['upper mid'].set_xlim(row-30,row+30)
#     axd['upper right'].set_ylim(column-4,column+4)
#     axd['upper mid'].set_ylim(column-4,column+4)
#     #scatter(np.arange(len(image[:,column])-2*wid)+wid, image[wid:-wid,column],
#     axd['lower right'].scatter(np.arange(len(streakavg)), streakavg,
#                 s=np.full(len(image[:,column]),10), color="black",
#                 marker="|")
#     axd['lower mid'].scatter(np.arange(len(streakavg)), streakavg,
#                 s=np.full(len(image[:,column]),10), color="black",
#                 marker="|")
#     axd['lower right'].set_xlim(row-9,row+9)
#     axd['lower mid'].set_xlim(row-30,row+30)
#     axd['upper right'].set_axis_off()
#     axd['upper mid'].set_axis_off()
#     # axd['upper left'].set_axis_off()
#     # axd['lower right'].set_xticks([])
#     axd['lower right'].set_yticks([])
#     axd['lower mid'].set_yticks([])
#     axd['upper left'].set_xticklabels([])
#     axd['upper left'].set_yticklabels([])
#     axd['upper left'].set_yticks([])
#     axd['upper left'].set_xticks([])

#     axd['lower left'].set_yticks([])
#     axd['lower left'].set_xticks(np.arange(0, len(image[:, column]), 500))
#     axd['lower left'].set_xticklabels(np.arange(0, len(image[:, column])/1000, 500/1000))
#     axd['lower right'].set_xticks(np.linspace(row-9,row+9, 3))
#     axd['lower mid'].set_xticks(np.linspace(row-30,row+30, 3))
#     # axd['lower right'].set_xticklabels([.87,.88,.89])
#     # axd['lower right'].set_xticklabels(np.arange(0, len(image[:, column])/1000, 500/1000))

#     # axd['lower left'].set_yticklabels([])
#     axd['lower left'].set_xlim(0,len(image[:,column]))
#     axd['lower left'].set_xlabel("Time (seconds)", fontsize=12)
#     axd['lower left'].set_ylabel("Flux (normalized)", fontsize=11)
#     axd['upper left'].set_ylabel("Streak image", fontsize=11)
#     axd['lower right'].set_xlabel("Time (ms)", fontsize=12)
#     axd['lower mid'].set_xlabel("Time (ms)", fontsize=12)
    
#     # fig.suptitle('column '+str(column)+" (b=transient, g=streak, r=background)")
#     #plt.savefig("/home/3155/timeseries_outputs/column "+str(column))
#     # if type(pathtosave)==str:
#     #     plt.savefig(pathtosave+"column "+str(column))
#     # plt.tight_layout()
#     fig.suptitle("Streak with possible transient #"+str(index)+" from folder "+direcname, fontsize=14)
#     plt.show()
#     plt.close()
    
    
# 

def get_sobel_ratio(stamp):
    '''
    input: stamp: 2d brightnesses array with candidate transient
    larger = less likely to be a cosmic ray
    returns sobel_ratio = np.max(Lx)/np.max(Ly) where x is columns axis and y is rows axis
    '''
    Lx=sobel(stamp,axis=1) #column axis
    Ly=sobel(stamp,axis=0) #rows axis
    sobel_ratio = np.max(Lx)/np.max(Ly) #we want this to be as large as possible
    return sobel_ratio

def get_stamp_from_df(index, direcname, df, norm=True):
    '''
    returns a 20x20 numpy array of brightnesses centered around the detection
    located at index of dataframe df
    note that the filepath to the science images is hardcoded currently
    norm=True has the image normalized (not the stamp, the entire image)
    norm=False has it in terms of counts
    '''
    fname = df.loc[index]["filename"]+"fits.fz"
    row = int(df.loc[index]["row_from_nn"])+10
    col = int(df.loc[index]["col_from_nn"])+10
    quad = int(df.loc[index]["quadrant"])
    # if direcname == "0820":
    fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"_minus_extras/"
    # else:
        # fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"/"
    path = fpath+fname
    hdul = fits.open(path)
    file1_array = hdul[quad].data
    if norm:
        normalized = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array)) #norm
        stamp = normalized[row-10:row+10, col-10:col+10]
        # plt.imshow(stamp)
        # plt.colorbar()
        return stamp
        background = get_image_background(normalized)
    stamp = file1_array[row-10:row+10, col-10:col+10]
    # plt.imshow(stamp)
    # plt.colorbar()
    background = get_image_background(file1_array)
    return stamp, background

def add_sobel_ratio(direcname, df):
    '''
    returns the dataframe with a new column of sobel ratios 
    see get_sobel_ratio(stamp)
    '''
    sobels = np.zeros(len(df))
    for index in range(len(df)):
        stamp = get_stamp_from_df(index, direcname, df, norm=False)
        sobel = get_sobel_ratio(stamp)
        sobels[index] = sobel
    df["sobel_ratio"] = sobels
    return df

# def plot_time_series(df, index, direcname):
#     '''
#     this plots a time series from a dataframe, making it and easy to diagnose
#     inputs: dataframe, index, directory where the images live
#     outputs: none, but plots.
#     specifications: both poisson and stdv error, no normalization
#     background is subtracted

#     '''
#     # index=2
#     # df=df_0820_short
#     # direcname = "0820"
#     wid=10
#     fname = df.iloc[index]["filename"]+"fits.fz"
#     row = int(df.iloc[index]["row_from_nn"])+10
#     col = int(df.iloc[index]["col_from_nn"])+10
#     fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"_minus_extras/"
#     path = fpath+fname
#     hdul=fits.open(path)
#     # print(path)
#     file1_array = hdul[1].data
#     norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array)) #norm
#     image = file1_array
#     plt.rcParams["figure.figsize"] = (10, 7)
#     plt.rcParams["image.interpolation"] = "nearest"
#     plt.rcParams["image.cmap"] = "gist_gray"
    
#     column = int(col)

#     #note: the scatter plots are shifted in the positive x direction by wid, 
#     #so that they line up with the images
    
#     gs_kw = dict(width_ratios=[3, 2, 1], height_ratios=[1,  2])
#     fig, axd = plt.subplot_mosaic([['upper left', 'upper mid', 'upper right'],
#                                ['lower left', 'lower mid','lower right']],
#                               gridspec_kw=gs_kw, figsize=(7.5, 3.5),
#                               layout="constrained")
    
#     background = get_image_background(image)

#         #subtracting background
#     imagenobackground = image - np.full(np.shape(image),background)
#     averages = np.average(imagenobackground, axis=0)
#     mid = int(col)
#     indices_to_check = [mid-6, mid-5, mid-4, mid-3, mid-2, mid-1, mid, mid+1, mid+2, mid+3, mid+4, mid+5] 
#     column_indices = []
#     for ind in indices_to_check:
#         if averages[ind]>.01:
#             column_indices.append(ind)
    
#     streaks = np.zeros(( len(column_indices), len(image[:,column])))
#     for i, inde in enumerate(column_indices):
#         streaks[i] = imagenobackground[:,inde]
#          # streaks[i] = image[:,inde]
#     streakavg = np.average(streaks, axis=0)
#     tot = np.sum(streaks, axis=0)
#     sqrt = np.sqrt(tot)
#     # fluxerr = np.divide(sqrt, len(column_indices))
#     fluxerr = sqrt

    
#     stderr = np.zeros(len(tot))
#     for i in range(len(stderr)):
#         if i>60 and i<(len(stderr)-60):
#             arr = np.concatenate((tot[i-60:i-10],tot[i+10:i+60]))
#             stderr[i]=np.std(arr)
                 
#     axd['lower left'].errorbar(np.arange(len(streakavg)), tot, yerr=fluxerr,
#                 color="black", ecolor='blue', elinewidth=1,linestyle='none',
#                 marker=".", markersize=1, alpha=.5, label="Poisson err")
#             # s=np.full(len(image[:,column])-2*wid,3), norm=mpl.colors.Normalize(vmin=0, vmax=6),
#             # marker="|", c=cimg.T[:,column-10], cmap="prism")
#     axd['lower left'].errorbar(np.arange(len(streakavg)), tot, yerr=stderr,
#                 color="black", ecolor='green', elinewidth=1,linestyle='none',
#                 marker=".", markersize=1, alpha=.5, label="Stdv err")
#     axd['lower left'].plot(np.arange(len(streakavg))[row-9:row+9], tot[row-9:row+9], 
#                 color="white", linestyle='none',
#                 marker=".", markersize=7, markeredgecolor='black', zorder=10)
    
    
#     axd['lower mid'].plot(np.arange(len(streakavg))[row-9:row+9], tot[row-9:row+9], 
#                 color="white", linestyle='none',
#                 marker=".", markersize=7, markeredgecolor='black', zorder=10)
    
#     axd['lower right'].plot(np.arange(len(streakavg))[row-9:row+9], tot[row-9:row+9], 
#                 color="white", linestyle='none',
#                 marker=".", markersize=7, markeredgecolor='black', zorder=10, label=None)
#     axd['upper left'].imshow(np.sqrt(image).T,aspect=70,cmap="gist_gray")
#     axd['upper left'].set_ylim(column-5,column+5)
#     axd['upper right'].imshow(np.sqrt(image).T,aspect=1.2,cmap="gist_gray")
#     axd['upper right'].set_xlim(row-9,row+9)
#     axd['upper right'].set_ylim(column-5,column+5)
    
#     axd['lower right'].errorbar(np.arange(len(streakavg)), tot, #np.arange(len(image[:,column])), image[:,column],
#                 color="black", #s=np.full(len(image[:,column]),10), 
#                 yerr=fluxerr,
#                 ecolor='blue', elinewidth=1, linestyle='none',
#                 marker=".", markersize=1, alpha=.5)
#     axd['lower right'].errorbar(np.arange(len(streakavg)), tot, yerr=stderr,
#                 color="black", ecolor='green', elinewidth=1,linestyle='none',
#                 marker=".", markersize=1, alpha=.5)
    
#     axd['lower right'].set_xlim(row-9,row+9)
#     axd['upper right'].set_axis_off()

#     axd['lower right'].set_yticks([])
#     axd['upper left'].set_xticklabels([])
#     axd['upper left'].set_yticklabels([])
#     axd['upper left'].set_yticks([])
#     axd['upper left'].set_xticks([])

#     axd['lower left'].set_xticks(np.arange(0, len(image[:, column]), 500))
#     axd['lower left'].set_xticklabels(np.arange(0, len(image[:, column])/1000, 500/1000))
#     axd['lower right'].set_xticks(np.linspace(row-9,row+9, 3))


#     axd['lower left'].set_xlim(0,len(image[:,column]))
#     axd['lower left'].set_xlabel("Time (seconds)", fontsize=12)
#     axd['lower left'].set_ylabel("Summed streak flux (counts)", fontsize=11)
#     axd['upper left'].set_ylabel("Streak image", fontsize=11)
#     axd['lower right'].set_xlabel("Time (ms)", fontsize=12)

    
#     axd['upper mid'].imshow(np.sqrt(image).T,aspect=2,cmap="gist_gray")
#     axd['upper mid'].set_xlim(row-30,row+30)
#     axd['upper mid'].set_ylim(column-5,column+5)  
#     axd['lower mid'].errorbar(np.arange(len(streakavg)), tot, yerr=fluxerr,
#                 color="black", ecolor='blue', elinewidth=1,linestyle='none',
#                 marker=".", markersize=1, alpha=.5)
#     # axd['lower mid'].set_xlim(row-30,row+30)
#     axd['lower mid'].set_xlim(row-100,row+100)
#     axd['lower mid'].set_xticks(np.linspace(row-100,row+100, 3))
#     axd['upper mid'].set_axis_off()    
#     axd['lower mid'].set_yticks([])
#     # axd['lower mid'].set_xticks(np.linspace(row-30,row+30, 3))
#     axd['lower mid'].set_xlabel("Time (ms)", fontsize=12)
#     axd['lower mid'].errorbar(np.arange(len(streakavg)), tot, yerr=stderr,
#                 color="black", ecolor='green', elinewidth=1,linestyle='none',
#                 marker=".", markersize=1, alpha=.5)
#     # axd['lower left'].legend([" ","Poisson err.", "Stdv err."])
#     axd['lower left'].legend()

#     fig.suptitle("Streak with possible transient #"+str(index)+" from folder "+direcname, fontsize=14)
#     plt.show()
#     plt.close()
    
# def inspect_df_time_series(reduced_df, full_df, direcname, start=0, stop=50):
#     num = stop-start
#     if len(reduced_df) < num:
#         stop = len(reduced_df) + start
#     for i in reduced_df["Unnamed: 0"][start:stop]:
#         print("image (below): " + full_df.iloc[i]["filename"])
#         print("transient candidate in column (streak) ", full_df.iloc[i]["col_from_nn"])
#         plot_time_series(full_df, i, direcname)
        
        
        
def reduce_edges(df, thr):
    '''
    if it's within 15 of an edge i don't trust it
    '''
    reduced = df[df["row_from_nn"]>thr]
    reduced = reduced[reduced["row_from_nn"]<(3080-10-thr)] #max is 3059
    reduced = reduced[reduced["col_from_nn"]>thr]
    reduced = reduced[reduced["col_from_nn"]<(3072-10-thr)] #max is 3051
    
    print("Number of transient candidates reduced from "+str(len(df))+" to "+str(len(reduced))+".")
    return reduced

def reduce_size(df, thr):
    reduced = df[df["detection_size"]>thr]
    print("Number of transient candidates reduced from "+str(len(df))+" to "+str(len(reduced))+".")
    return reduced

def reduce_cr_flag(df, thr):
    '''
    SHOULD be less than thr
    '''
    reduced = df[df["cosmic_ray_flag"]<thr]
    print("Number of transient candidates reduced from "+str(len(df))+" to "+str(len(reduced))+".")
    return reduced

def reduce_sobel(df, thr):
    '''
    cuts out transient candidates where their sobel ratio is less than thr
    returns dataframe of candidates with sobel ration greater than thr
    '''
    reduced = df[df["sobel_ratio"]>thr]
    print("Number of transient candidates reduced from "+str(len(df))+" to "+str(len(reduced))+".")
    return reduced

def reduce_all(df, direc, edge_thr=15, size_thr=4, cr_thr=.1, sobel_thr=4):
    '''
    Takes a dataframe of transient candidates and reduces it:
    Eliminating canditates too close to edges (within %.0f)..."%edge_thr
    "Eliminating canditates that are not greater than %.0f pixels..."%size_thr)
    "Eliminating canditates that are not on streaks (threshold: %.2f)"%cr_thr)
    "Eliminating canditates with more edges across the streak than down the streak (thr: %.0f)"%sobel_thr
    And if there is no column for sobel ratio in the dataframe, calculating that column as well.
    Returns new daraframe. also lengths of cuts

    '''
    lengths = np.zeros(5)
    lengths[0] = len(df)
    print("Initial dataframe has %.0f transient candidates."%len(df))
    print("Eliminating canditates too close to edges (within %.0f)..."%edge_thr)
    reduced = reduce_edges(df, edge_thr)
    lengths[1] = len(reduced)
    print("Eliminating canditates that are not greater than %.0f pixels..."%size_thr)
    reduced = reduce_size(reduced, size_thr)
    lengths[2] = len(reduced)
    print("Eliminating canditates that are not on streaks (threshold: %.2f)"%cr_thr)
    reduced = reduce_cr_flag(reduced, cr_thr)
    lengths[3] = len(reduced)
    if "sobel_ratio" not in list(df):
        print("Calculating sobel filter ratios...")
        reduced = add_sobel_ratio(direc, reduced)
    print("Eliminating canditates with more edges across the streak than down the streak (thr: %.0f)"%sobel_thr)
    reduced = reduce_sobel(reduced, sobel_thr)
    lengths[4] = len(reduced)
    return reduced, lengths


def reduce_number_in_file(df, max=1):
    '''
    takes a dataframe of transient candidates, 
    eliminates all candidates where there is more than max (default 1) per image
    returns new dataframe of candidates that are solo(ish) on their images
    
    '''
    good_indices = []
    unique, counts = np.unique(df['filename'], return_counts=True)
    for i, file in enumerate(unique[counts<=max]):
        if i==0:
            ind = df.index[df['filename'] == file].tolist() #[0]
            # print(ind)
            good_indices.append(ind)
            solos_df = pd.DataFrame(df[df['filename'] == file])
            solos_df
        else:
            ind = df.index[df['filename'] == file].tolist() #[0]
            # print(ind)
            good_indices.append(ind)
            solos_df = pd.concat([solos_df, df[df['filename'] == file]], ignore_index=True)
    return solos_df

# def get_stdv_ratio(df, index, direcname):
#     '''
#     NOTE: ONLY woRKS FOR HDUL1 RIGHT NOW
#     gets the ratio of the midpoint of the detection to the average of the streak, 
#     in units of standard deviations

#     '''
#     # index=2
#     # df=df_0820_short
#     # direcname = "0820"
#     wid=10
#     fname = df.iloc[index]["filename"]+"fits.fz"
#     row = int(df.iloc[index]["row_from_nn"])+10
#     col = int(df.iloc[index]["col_from_nn"])+10
#     fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"_minus_extras/"
#     path = fpath+fname
#     hdul=fits.open(path)
#     # print(path)
#     file1_array = hdul[1].data
#     norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array)) #norm
#     image = file1_array

#     column = int(col)


    
#     background = get_image_background(image)

#         #subtracting background
#     imagenobackground = image - np.full(np.shape(image),background)
#     averages = np.average(imagenobackground, axis=0)
#     mid = int(col)
#     indices_to_check = [mid-6, mid-5, mid-4, mid-3, mid-2, mid-1, mid, mid+1, mid+2, mid+3, mid+4, mid+5] 
#     column_indices = []
#     for ind in indices_to_check:
#         if averages[ind]>.01:
#             column_indices.append(ind)
    
#     streaks = np.zeros(( len(column_indices), len(image[:,column])))
#     for i, inde in enumerate(column_indices):
#         streaks[i] = imagenobackground[:,inde]
#          # streaks[i] = image[:,inde]
#     streakavg = np.average(streaks, axis=0)
#     tot = np.sum(streaks, axis=0)
#     sqrt = np.sqrt(tot)
#     # fluxerr = np.divide(sqrt, len(column_indices))
#     fluxerr = sqrt

#     stdtot = np.std(tot)
#     centerbrightness = tot[mid]
#     avg = np.mean(tot)
#     diff = centerbrightness - avg
#     diffinstd = diff/stdtot
#     return diffinstd




def inspect_df_with_quadrant(df, direcname):
    '''
    plots the stamps corresponding to detections in a dataframe
    no limit, so put in df[:100] or something probably to save time
    '''
    wid = 10
    figcols = 5
    figrows = 1+len(df)//figcols
    fig, ax = plt.subplots(figrows, figcols, figsize=(figcols*1.8, figrows*1.8))
    for i in range(figrows):
        for j in range(figcols):
            index = figcols*i+j
            if index<len(df):
                fname = df.iloc[index]["filename"]+"fits.fz"
                row = int(df.iloc[index]["row_from_nn"])
                col = int(df.iloc[index]["col_from_nn"])
                quad = int(df.iloc[index]["quadrant"])
                fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"_minus_extras/"
                path = fpath+fname
                hdul=fits.open(path)
                file1_array = hdul[quad].data
                norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array))
                # ax[i, j].imshow(norm[col+wid-10:col+wid+10, row-10+wid:row+10+wid])
                ax[i, j].imshow(np.sqrt(norm[row-10+wid:row+10+wid, col+wid-10:col+wid+10]))
                ax[i, j].set_axis_off()
                # ax[i, j].set_title(fname[26:-8])
                ax[i, j].set_title(str(index))
    plt.tight_layout()
    plt.show()
    
    
def get_tot(df, index, half_aperture=2, tracking=True, fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/1020_minus_extras/"):
    '''
    returns tot, maxcols: an array that is the flux of the star streak
    given an aperture that is halfaperture*2+1
    and if tracking=true, tracks when the streak posittion fluctuates
    oif tracking==false, goes based on average center of streak
    note: uses loc instead of iloc
    '''
    wideval = half_aperture
    wid=10
    fname = df.loc[index]["filename"]+"fits.fz"
    row = int(df.loc[index]["row_from_nn"])+10 #we add 10 here bc that's the cnn offset of returned values
    col = int(df.loc[index]["col_from_nn"])+10
    quad= int(df.loc[index]["quadrant"])
    column = int(col)
        
    # print(fpath)
    # print(fname)
    path = fpath+fname
    hdul=fits.open(path)
    # print(path)
    file1_array = hdul[quad].data
    # norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array)) #norm
    image = file1_array

    

    background = get_image_background(image)

    imagenobackground = image - np.full(np.shape(image),background)

    if tracking==False:
        zeropoint = col-7
        averages = np.average(imagenobackground[row-10:row+10, zeropoint:zeropoint+14], axis=0)
        maxind = np.argmax(averages)
        tot2 = np.zeros(imagenobackground.shape[0])
        for r in range(0, imagenobackground.shape[0]): #this should be each row, probably
            maxcol = zeropoint + maxind
            combined = imagenobackground[r, maxcol]
            for i in range(1,half_aperture+1):
                combined+=imagenobackground[r, maxcol-i]
                combined+=imagenobackground[r, maxcol+i]
            tot2[r] = combined
        return tot2, np.full(len(tot2), maxcol)
    
    #so we need to track the whole streak
    #for each row, need to choose a column index
    tot = np.zeros(imagenobackground.shape[0])
    maxcols = np.zeros(imagenobackground.shape[0])
    zeropoint = col-10
    for r in range(0, imagenobackground.shape[0]): #this should be each row, probably
        maxcol = np.argmax(imagenobackground[r, zeropoint:zeropoint+20])+ zeropoint
        maxcols[r] = maxcol
        combined = imagenobackground[r, maxcol]
        for i in range(1,half_aperture+1):
            combined+=imagenobackground[r, maxcol-i]
            combined+=imagenobackground[r, maxcol+i]
        # combined = imagenobackground[r, maxcol-2] + imagenobackground[r, maxcol-1] + imagenobackground[r, maxcol] + imagenobackground[r, maxcol+1] + imagenobackground[r, maxcol+2]
        # print(combined)
        tot[r] = combined
        

    return tot, maxcols





    
def get_stdvs(tot, do_all=True, df=None, index=None):
    '''
    takes tot
    do_all: if true, calculates the stdv for every point
    if false, calculates the stdv around the detection
    if do_all=false, must include df and index
    '''
    if do_all==True:
        stdvs = np.zeros(len(tot))
        for i in range(len(tot)):
            if i>60 and i<(len(tot)-60):
                arr = np.concatenate((tot[i-60:i-10],tot[i+10:i+60]))
                stdvs[i]=np.std(arr)
    else: 
        row = int(df.loc[index]["row_from_nn"])+10
        arr = np.concatenate((tot[row-60:row-10],tot[row+10:row+60]))
        stdvs = np.std(arr)
    return stdvs


def plot_time_series_again_aperture(df, index, direcname='1020', half_aperture=2, tracking=True, title=None, plot_tracking=False, ylims=0):
    '''
    this plots a time series and image from a dataframe, 
    showing which pixels are included
    inputs: dataframe, index, directory where the images live
    aperture is 2*half_aperture+1
    tracking can be True (tracks with brightest pixel)
    or False (constant streak center pixel)
    outputs: none, but plots.
    NOTE: now uses loc intead of iloc, so it corresponds to the actual index!!
    '''
    # index=2
    # df=df_0820_short
    # direcname = "0820"
    wid=10
    fname = df.loc[index]["filename"]+"fits.fz"
    row = int(df.loc[index]["row_from_nn"])+10
    col = int(df.loc[index]["col_from_nn"])+10
    quad = int(df.loc[index]["quadrant"])

    # print(testindex)
    # print(df.iloc[testindex]["filename"])


    # print("indices to check")
    # print(same_img_indices)

        
    fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"_minus_extras/"
    path = fpath+fname
    hdul=fits.open(path)
    # print(path)
    file1_array = hdul[quad].data
    norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array)) #norm
    image = file1_array
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["image.interpolation"] = "nearest"
    plt.rcParams["image.cmap"] = "gist_gray"
    # plt.imshow(norm)
    # plt.xlim(1070,1170)
    # plt.ylim(100,120)
    
    column = int(col)

    #note: the scatter plots are shifted in the positive x direction by wid, 
    #so that they line up with the images
    
    gs_kw = dict(width_ratios=[2,2], height_ratios=[1,  2])
    fig, axd = plt.subplot_mosaic([['upper left', 'upper mid'],
                               ['lower left', 'lower mid']],
                              gridspec_kw=gs_kw, figsize=(7.5, 3.5),
                              layout="constrained")
    
    background = get_image_background(image)

   

    imagenobackground = image - np.full(np.shape(image),background)
    
    tot, maxcols = get_tot(df, index,  half_aperture, tracking)
    
    if plot_tracking==True:
        axd['upper left'].plot(maxcols+half_aperture+1, color='yellow')
        axd['upper left'].plot(maxcols-half_aperture, color='yellow')        
        axd['upper mid'].plot(maxcols+half_aperture+1, color='yellow')
        axd['upper mid'].plot(maxcols-half_aperture, color='yellow')
    if ylims!=0:
        lower=ylims[0]
        upper=ylims[1]
        axd['lower left'].set_ylim(lower, upper)#(7850,14000)
        axd['lower mid'].set_ylim(lower, upper)
    
    
    sqrt = np.sqrt(tot/6.2+8.5**2) #6.2 electrons per photon, read noise is 8.5
    # fluxerr = np.divide(sqrt, len(column_indices))
    fluxerr = sqrt
    streakavg = tot #i don't use this any more
    


    
    stderr = np.zeros(len(tot))
    for i in range(len(stderr)):
        if i>60 and i<(len(stderr)-60):
            arr = np.concatenate((tot[i-60:i-10],tot[i+10:i+60]))
            stderr[i]=np.std(arr)
            
    axd['lower left'].plot(np.arange(len(streakavg))[row-9:row+9], tot[row-9:row+9], 
                color="white", linestyle='none',
                marker=".", markersize=7, markeredgecolor='black', zorder=10)
    
    axd['lower mid'].plot(np.arange(len(streakavg))[row-9:row+9], tot[row-9:row+9], 
                color="white", linestyle='none',
                marker=".", markersize=7, markeredgecolor='black', zorder=10)
    
                 
    axd['lower left'].errorbar(np.arange(len(streakavg)), tot, yerr=fluxerr,
                color="black", ecolor='black', elinewidth=1,linestyle='none',
                marker=".", markersize=1, alpha=.4, label="Shot Noise")

    axd['lower left'].errorbar(np.arange(len(streakavg)), tot, yerr=stderr,
                color="black", ecolor='SteelBlue', elinewidth=1,linestyle='none',
                marker=".", markersize=1, alpha=.2, label=r"$\sigma_{Flux}$")
    
    
    axd['lower mid'].errorbar(np.arange(len(streakavg)), tot, yerr=fluxerr,
                color="black", ecolor='black', elinewidth=1, linestyle='none',
                marker=".", markersize=1, alpha=.4)
    
    axd['lower mid'].errorbar(np.arange(len(streakavg)), tot, yerr=stderr,
                color="black", ecolor='SteelBlue', elinewidth=1,linestyle='none',
                marker=".", markersize=1, alpha=.2)
                 
#     axd['lower left'].fill_between(np.arange(len(streakavg)), tot+fluxerr/2, tot-fluxerr/2, 
#                                    alpha=.5, label="Poisson error") 
#                 # color="black", ecolor='blue', elinewidth=1,linestyle='none',
#                 # marker=".", markersize=1, alpha=.5, label="Poisson error")

#     axd['lower left'].fill_between(np.arange(len(streakavg)), tot+stderr/2, tot-stderr/2, 
#                                    alpha=.3, label="Stdv error")
# axd['lower left'].fill_between(np.arange(len(streakavg)), tot, yerr=stderr,
#                 color="black", ecolor='green', elinewidth=1,linestyle='none',
#                 marker=".", markersize=1, alpha=.5, label="Stdv error")

    

    axd['upper left'].imshow(np.sqrt(image).T,aspect=20,cmap="gist_gray")
    axd['upper left'].set_ylim(column-20,column+20)
 


    axd['upper left'].set_xticklabels([])
    axd['upper left'].set_yticklabels([])
    axd['upper left'].set_yticks([])
    axd['upper left'].set_xticks([])

    # axd['lower left'].set_xti3cks(np.linspace(0, len(image[:, column]), 6))
    
    # axd['lower left'].set_xticklabels(np.round(np.linspace(0, len(image[:, column])*10/6000, 6*10/6000)), fontsize=12)
    axd['lower left'].set_xticks(np.linspace(0, 3080*10/11, 10))
    axd['lower left'].set_xticklabels(np.round(np.linspace(0, 9, 10, dtype='int')), fontsize=13)



    axd['lower left'].set_xlim(0,len(image[:,column]))
    axd['lower left'].set_xlabel("Time (seconds)", fontsize=12)
    axd['lower left'].set_ylabel("Flux (counts)", fontsize=11)
    axd['upper left'].set_ylabel("Image", fontsize=11)

    
    axd['upper mid'].imshow(np.sqrt(image).T,aspect=.3,cmap="gist_gray")
    axd['upper mid'].set_xlim(row-20,row+20)
    axd['upper mid'].set_ylim(column-20,column+20)  

    # axd['lower mid'].set_xlim(row-30,row+30)
    axd['lower mid'].set_xlim(row-20,row+20)
    # axd['lower mid'].set_xticks(np.linspace(row-20,row+20, 3))
    axd['lower mid'].set_xticks(np.linspace(row-16,row+16, 3), labels=np.linspace((row-16)*10/3,(row+16)*10/3, 3, dtype='int'), fontsize=13)

    axd['upper mid'].set_axis_off()    
    axd['lower mid'].set_yticks([])
    # axd['lower mid'].set_xticks(np.linspace(row-30,row+30, 3))
    axd['lower mid'].set_xlabel("Time (milliseconds)", fontsize=12)

    # axd['lower left'].legend([" ","Poisson err.", "Stdv err."])
    axd['lower left'].legend()
    if title==None:
        fig.suptitle("Possible Transient # "+str(index), fontsize=24)
    else:
        fig.suptitle(title, fontsize=24)

    plt.show()
    plt.close()

def get_milliseconds_from_khz(freq):
    '''
    put in frequency in khz, get:
    mspercycle, secpercycle, pixpercycle
    '''
    #we have cycles/millisecond
    mspercycle = 1/freqtot
    secpercycle = mspercycle/1000
    pixpercycle = mspercycle*3.08/10
    return mspercycle, secpercycle, pixpercycle


def add_snr(df, direcname, half_aperture, tracking):
    '''
    returns the dataframe with a new column of stdv ratios 
    see get_sobel_ratio(stamp) and get_stdv_ratio
    '''
    ratios = np.zeros(len(df))
    for index in df.index:
        # print(index)
        tot, maxcols = get_tot(df, index, half_aperture, tracking)
        stdv = get_stdvs(tot, do_all=False, df=df, index=index)
        avg = np.average(tot)
        row = int(df.loc[index]["row_from_nn"])+10
        signal = tot[row] - avg
        ratios[index] = signal/stdv
    df["snr"] = ratios
    return df


def get_tot_streak(image, col, tracking=True, half_aperture=2):
    '''
    '''
    # row = int(df.iloc[index]["row_from_nn"])+10 #we add 10 here bc that's the cnn offset of returned values
    # col = int(df.iloc[index]["col_from_nn"])+10
    # col = int(peaks[peakindex])
    row = 1500 #for calculating avg
    background = get_image_background(image)
    imagenobackground = image - np.full(np.shape(image),background)
    if tracking==False:
        zeropoint = col-7
        averages = np.average(imagenobackground[row-50:row+50, zeropoint:zeropoint+14], axis=0)
        maxind = np.argmax(averages)
        tot2 = np.zeros(imagenobackground.shape[0])
        for r in range(0, imagenobackground.shape[0]): #this should be each row, probably
            maxcol = zeropoint + maxind
            combined = imagenobackground[r, maxcol]
            for i in range(1,half_aperture+1):
                combined+=imagenobackground[r, maxcol-i]
                # combined+=imagenobackground[r, maxcol+i]
            tot2[r] = combined
        return tot2

    #so we need to track the whole streak
    #for each row, need to choose a column index
    tot = np.zeros(imagenobackground.shape[0])
    maxcols = np.zeros(imagenobackground.shape[0])
    zeropoint = col-10
    for r in range(0, imagenobackground.shape[0]): #this should be each row, probably
        maxcol = np.argmax(imagenobackground[r, zeropoint:zeropoint+20])+ zeropoint
        maxcols[r] = maxcol
        combined = imagenobackground[r, maxcol]
        for i in range(1,half_aperture+1):
            combined+=imagenobackground[r, maxcol-i]
            combined+=imagenobackground[r, maxcol+i]
        # combined = imagenobackground[r, maxcol-2] + imagenobackground[r, maxcol-1] + imagenobackground[r, maxcol] + imagenobackground[r, maxcol+1] + imagenobackground[r, maxcol+2]
        # print(combined)
        tot[r] = combined


    

    return tot, maxcols