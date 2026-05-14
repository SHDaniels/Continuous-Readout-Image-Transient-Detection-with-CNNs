import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import sobel
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import gc  # garbage collector
import scipy.ndimage as snm
import precision_recall as pr
import get_postage_stamps as gpsf
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns


def getdf(direcname):
    """
    deprecated
    """
    fpath_extractions = "/lustre/lrspec/ZTF_data/science_extractions/" + direcname + "/"
    dfname = fpath_extractions + "df_detections_" + str(direcname)
    print(dfname)
    df = pd.read_csv(dfname)
    return df


def df_shorten(df, size=None, bgmax=None, brlim=None, num_desired=None):
    dfleninit = len(df)
    if size != None:
        df = df[df["detection_size"] > size]
    if bgmax != None:
        df = df[df["cosmic_ray_flag"] < bgmax]
    if brlim != None:
        df = df[df["max brightness"] > brlim]
    if num_desired != None:
        print("didn't set up num desired")
    print("Dataframe shortened from " + str(dfleninit) + " to " + str(len(df)) + ".")
    return df


def inspect_df(df, direcname, fpath=None, extension = "fits.fz", normalize=False, star=False):
    wid = 10
    figcols = 5
    figrows = 1 + len(df) // figcols
    fig, ax = plt.subplots(figrows, figcols, figsize=(figcols * 1.8, figrows * 1.8))
    for i in range(figrows):
        for j in range(figcols):
            index = figcols * i + j
            if index < len(df):
                fname = df.iloc[index]["filename"] + extension
                row = int(df.iloc[index]["row_from_nn"])
                col = int(df.iloc[index]["col_from_nn"])
                
                if fpath==None:
                    fpath = (
                        "/lustre/lrspec/ZTF_data/2022/1020/"
                    )
                path = fpath + fname
                if extension == "fits.fz":
                    hdul = fits.open(path)
                    file1_array = hdul[1].data
                else:
                    file1_array = np.load(path)
                if normalize:
                    norm = (file1_array - np.min(file1_array)) / (
                        np.max(file1_array) - np.min(file1_array)
                    )
                else:
                    norm=file1_array
                # ax[i, j].imshow(norm[col+wid-10:col+wid+10, row-10+wid:row+10+wid])
                ax[i, j].imshow(
                    np.sqrt(
                        norm[
                            row - 10 + wid : row + 10 + wid,
                            col + wid - 10 : col + wid + 10,
                        ]
                    )
                )
                ax[i, j].set_axis_off()
                # ax[i, j].set_title(fname[26:-8])
                ax[i, j].set_title(df.iloc[index]["Unnamed: 0"])
    plt.tight_layout()
    plt.show()
    # ax[i, j].set_ylim(col+wid-10,col+wid+10)
    # ax[i, j].set_xlim(row-10+wid,row+10+wid)
    # print(path)


def get_image_background(image):
    """
    this function gets the background value of an array of various brightnesses
    and iteratively clips the maximum at a 3 standard deviation level to get the background value
    input: array of any dimensions
    return: background value (float)
    """

    change = 100
    newnorm = image
    while np.abs(change) > 0.0001:
        stdv = np.std(newnorm)
        norm_mean = np.mean(newnorm)
        newnorm = np.clip(
            newnorm, 0, norm_mean + 3 * stdv
        )  # check - is this the right cutoff
        newnorm_mean = np.mean(newnorm)
        change = newnorm_mean - norm_mean
        background = newnorm_mean
    return background




def get_sobel_ratio(stamp):
    """
    input: stamp: 2d brightnesses array with candidate transient
    larger = less likely to be a cosmic ray
    returns sobel_ratio = np.max(Lx)/np.max(Ly) where x is columns axis and y is rows axis
    """
    Lx = sobel(stamp, axis=1)  # column axis
    Ly = sobel(stamp, axis=0)  # rows axis
    sobel_ratio = np.max(Lx) / np.max(Ly)  # we want this to be as large as possible
    return sobel_ratio, Lx, Ly


def get_stamp_from_df(index, df, direcname="2022/1020/", norm=True, quad=None, addrow=0, addcol=0, iloc=False,imgsize=10):
    """
    returns a 20x20 numpy array of brightnesses centered around the detection
    located at index of dataframe df
    note that the filepath to the science images is hardcoded currently
    norm=True has the image normalized (not the stamp, the entire image)
    norm=False has it in terms of counts
    """
    if iloc==False:
        fname = df.loc[index]["filename"] + "fits.fz"
        row = int(df.loc[index]["row_from_nn"]) + 10 + addrow
        col = int(df.loc[index]["col_from_nn"]) + 10 + addcol
        if quad==None:
            quad = int(df.loc[index]["quadrant"])
    elif iloc==True:
        fname = df.iloc[index]["filename"] + "fits.fz"
        row = int(df.iloc[index]["row_from_nn"]) + 10 + addrow
        col = int(df.iloc[index]["col_from_nn"]) + 10 + addcol
        if quad==None:
            quad = int(df.iloc[index]["quadrant"])        
    # if direcname == "0820":
    fpath = (
        "/lustre/lrspec/ZTF_data/"+direcname
    )
    # else:
    # fpath = "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"+direcname+"/"
    path = fpath + fname
    hdul = fits.open(path)
    file1_array = hdul[quad].data
    if norm:
        normalized = (file1_array - np.min(file1_array)) / (
            np.max(file1_array) - np.min(file1_array)
        )  # norm
        stamp = normalized[row - imgsize : row + imgsize, col - imgsize : col + imgsize]
        # plt.imshow(stamp)
        # plt.colorbar()
        background = get_image_background(normalized)
        return stamp, background
        
    stamp = file1_array[row - 10 : row + 10, col - 10 : col + 10]
    # plt.imshow(stamp)
    # plt.colorbar()
    background = get_image_background(file1_array)
    return stamp, background







def reduce_edges(df, thr):
    """
    if it's within 15 of an edge i don't trust it
    """
    reduced = df[df["row_from_nn"] > thr]
    reduced = reduced[reduced["row_from_nn"] < (3080 - 10 - thr)]  # max is 3059
    reduced = reduced[reduced["col_from_nn"] > thr]
    reduced = reduced[reduced["col_from_nn"] < (3072 - 10 - thr)]  # max is 3051

    print(
        "Number of transient candidates reduced from "
        + str(len(df))
        + " to "
        + str(len(reduced))
        + "."
    )
    return reduced


def reduce_size(df, thr):
    reduced = df[df["detection_size"] > thr]
    print(
        "Number of transient candidates reduced from "
        + str(len(df))
        + " to "
        + str(len(reduced))
        + "."
    )
    return reduced

def reduce_max_3(df, imagenobg, thr=300):
    drop_indices = []
    for index in df.index.to_numpy():
        row = int(df.loc[index]["row_from_nn"]) + 10
        col = int(df.loc[index]["col_from_nn"]) + 10
        littlerstamp = imagenobg[row - 4 : row + 4, col - 4 : col + 4]

        if np.max(littlerstamp) < thr:
            drop_indices.append(index)
    newdf = df.drop(drop_indices)
    return newdf


def reduce_cr_flag(df, thr):
    """
    SHOULD be less than thr
    """
    reduced = df[df["cosmic_ray_flag"] < thr]
    print(
        "Number of transient candidates reduced from "
        + str(len(df))
        + " to "
        + str(len(reduced))
        + "."
    )
    return reduced


def reduce_sobel(df, thr):
    """
    cuts out transient candidates where their sobel ratio is less than thr
    returns dataframe of candidates with sobel ration greater than thr
    """
    reduced = df[df["sobel_ratio"] > thr]
    print(
        "Number of transient candidates reduced from "
        + str(len(df))
        + " to "
        + str(len(reduced))
        + "."
    )
    return reduced


def reduce_all(df, direc, edge_thr=15, size_thr=4, cr_thr=0.1, sobel_thr=4):
    """
    Takes a dataframe of transient candidates and reduces it:
    Eliminating canditates too close to edges (within %.0f)..."%edge_thr
    "Eliminating canditates that are not greater than %.0f pixels..."%size_thr)
    "Eliminating canditates that are not on streaks (threshold: %.2f)"%cr_thr)
    "Eliminating canditates with more edges across the streak than down the streak (thr: %.0f)"%sobel_thr
    And if there is no column for sobel ratio in the dataframe, calculating that column as well.
    Returns new daraframe. also lengths of cuts

    """
    lengths = np.zeros(5)
    lengths[0] = len(df)
    print("Initial dataframe has %.0f transient candidates." % len(df))
    print("Eliminating canditates too close to edges (within %.0f)..." % edge_thr)
    reduced = reduce_edges(df, edge_thr)
    lengths[1] = len(reduced)
    print("Eliminating canditates that are not greater than %.0f pixels..." % size_thr)
    reduced = reduce_size(reduced, size_thr)
    lengths[2] = len(reduced)
    print("Eliminating canditates that are not on streaks (threshold: %.2f)" % cr_thr)
    reduced = reduce_cr_flag(reduced, cr_thr)
    lengths[3] = len(reduced)
    if "sobel_ratio" not in list(df):
        print("Calculating sobel filter ratios...")
        reduced = add_sobel_ratio(direc, reduced)
    print(
        "Eliminating canditates with more edges across the streak than down the streak (thr: %.0f)"
        % sobel_thr
    )
    reduced = reduce_sobel(reduced, sobel_thr)
    lengths[4] = len(reduced)
    return reduced, lengths


def reduce_number_in_file(df, max=1):
    """
    takes a dataframe of transient candidates,
    eliminates all candidates where there is more than max (default 1) per image
    returns new dataframe of candidates that are solo(ish) on their images

    """
    good_indices = []
    unique, counts = np.unique(df["filename"], return_counts=True)
    for i, file in enumerate(unique[counts <= max]):
        if i == 0:
            ind = df.index[df["filename"] == file].tolist()  # [0]
            # print(ind)
            good_indices.append(ind)
            solos_df = pd.DataFrame(df[df["filename"] == file])
            solos_df
        else:
            ind = df.index[df["filename"] == file].tolist()  # [0]
            # print(ind)
            good_indices.append(ind)
            solos_df = pd.concat(
                [solos_df, df[df["filename"] == file]], ignore_index=True
            )
    return solos_df




def inspect_df_with_quadrant(df, imgsize=10, star=False, clim=None, titlecol=None):
    """
    plots the stamps corresponding to detections in a dataframe
    no limit, so put in df[:100] or something probably to save time
    """
    wid = 10
    figcols = 5
    figrows = 1 + len(df) // figcols
    fig, ax = plt.subplots(figrows, figcols, figsize=(figcols * 1.8, figrows * 1.8))
    for i in range(figrows):
        for j in range(figcols):
            index = figcols * i + j
            if index < len(df):
                fname = df.iloc[index]["filename"] + "fits.fz"
                row = int(df.iloc[index]["row_from_nn"])
                col = int(df.iloc[index]["col_from_nn"])
                quad = int(df.iloc[index]["quadrant"])
                fpath = (
                    "/lustre/lrspec/ZTF_data/2022/1020/"
                )
                path = fpath + fname
                hdul = fits.open(path)
                file1_array = hdul[quad].data
                norm = (file1_array - np.min(file1_array)) / (
                    np.max(file1_array) - np.min(file1_array)
                )
                # ax[i, j].imshow(norm[col+wid-10:col+wid+10, row-10+wid:row+10+wid])
                ax[i, j].imshow(
                    np.sqrt(
                        norm[
                            row - imgsize + wid : row + imgsize + wid,
                            col + wid - imgsize : col + wid + imgsize,
                        ]
                    ), vmax=clim
                )
                if star==True:
                    starpos = df.iloc[index]['closest_x']
                    ax[i, j].scatter(starpos- (col - imgsize  + wid), 10)
                
                ax[i, j].set_axis_off()
                # ax[i, j].set_title(fname[26:-8])
                if titlecol==None:
                    ax[i, j].set_title(str(index))
                else:
                    ax[i, j].set_title(round(df.iloc[index][titlecol]))

    plt.tight_layout()
    plt.show()


def get_tot(
    df,
    index,
    half_aperture=2,
    tracking=True,
    fpath="/lustre/lrspec/ZTF_data/2022/1020/",
):
    """
    returns tot, maxcols: an array that is the flux of the star streak
    given an aperture that is halfaperture*2+1
    and if tracking=true, tracks when the streak posittion fluctuates
    oif tracking==false, goes based on average center of streak
    note: uses loc instead of iloc
    """
    wideval = half_aperture
    wid = 10
    fname = df.loc[index]["filename"] + "fits.fz"
    row = (
        int(df.loc[index]["row_from_nn"]) + 10
    )  # we add 10 here bc that's the cnn offset of returned values
    col = int(df.loc[index]["col_from_nn"]) + 10
    quad = int(df.loc[index]["quadrant"])
    column = int(col)

    # print(fpath)
    # print(fname)
    path = fpath + fname
    hdul = fits.open(path)
    # print(path)
    file1_array = hdul[quad].data
    # norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array)) #norm
    image = file1_array

    background = get_image_background(image)

    imagenobackground = image - np.full(np.shape(image), background)

    if tracking == False:
        zeropoint = col - 7
        averages = np.average(
            imagenobackground[row - 10 : row + 10, zeropoint : zeropoint + 14], axis=0
        )
        maxind = np.argmax(averages)
        tot2 = np.zeros(imagenobackground.shape[0])
        for r in range(
            0, imagenobackground.shape[0]
        ):  # this should be each row, probably
            maxcol = zeropoint + maxind
            combined = imagenobackground[r, maxcol]
            for i in range(1, half_aperture + 1):
                combined += imagenobackground[r, maxcol - i]
                combined += imagenobackground[r, maxcol + i]
            tot2[r] = combined
        return tot2, np.full(len(tot2), maxcol)

    # so we need to track the whole streak
    # for each row, need to choose a column index
    tot = np.zeros(imagenobackground.shape[0])
    maxcols = np.zeros(imagenobackground.shape[0])
    zeropoint = col - 10
    for r in range(0, imagenobackground.shape[0]):  # this should be each row, probably
        maxcol = np.argmax(imagenobackground[r, zeropoint : zeropoint + 20]) + zeropoint
        maxcols[r] = maxcol
        combined = imagenobackground[r, maxcol]
        for i in range(1, half_aperture + 1):
            combined += imagenobackground[r, maxcol - i]
            combined += imagenobackground[r, maxcol + i]
        # combined = imagenobackground[r, maxcol-2] + imagenobackground[r, maxcol-1] + imagenobackground[r, maxcol] + imagenobackground[r, maxcol+1] + imagenobackground[r, maxcol+2]
        # print(combined)
        tot[r] = combined

    return tot, maxcols


def get_stdvs(tot, do_all=True, df=None, index=None):
    """
    takes tot
    do_all: if true, calculates the stdv for every point
    if false, calculates the stdv around the detection
    if do_all=false, must include df and index
    """
    if do_all == True:
        stdvs = np.zeros(len(tot))
        for i in range(len(tot)):
            if i > 60 and i < (len(tot) - 60):
                arr = np.concatenate((tot[i - 60 : i - 10], tot[i + 10 : i + 60]))
                stdvs[i] = np.std(arr)
    else:
        row = int(df.loc[index]["row_from_nn"]) + 10
        arr = np.concatenate((tot[row - 60 : row - 10], tot[row + 10 : row + 60]))
        stdvs = np.std(arr)
    return stdvs


def plot_time_series_again_aperture(
    df,
    index,
    direcname="1020",
    half_aperture=2,
    tracking=True,
    title=None,
    plot_tracking=False,
    ylims=0, csv=True, lc_filename=None, fname=None,row=None,col=None,quad=None, lcfpath=None
):
    """
    this plots a time series and image from a dataframe,
    showing which pixels are included
    inputs: dataframe, index, directory where the images live
    aperture is 2*half_aperture+1
    tracking can be True (tracks with brightest pixel)
    or False (constant streak center pixel)
    outputs: none, but plots.
    NOTE: now uses loc intead of iloc, so it corresponds to the actual index!!
    """
    # index=2
    # df=df_0820_short
    # direcname = "0820"
    wid = 10
    if fname==None:
        fname = df.loc[index]["filename"] + "fits.fz"
    if row==None:
        row = int(df.loc[index]["row_from_nn"]) + 10
    if col==None:
        col = int(df.loc[index]["col_from_nn"]) + 10
    if quad==None:
        quad = int(df.loc[index]["quadrant"])

    # print(testindex)
    # print(df.iloc[testindex]["filename"])

    # print("indices to check")
    # print(same_img_indices)

    # fpath = (
    #     "/lustre/lrspec/ZTF_data/science_extractions/to_extract/"
    #     + direcname
    #     + "_minus_extras/"
    # )
    fpath = (
        "/lustre/lrspec/ZTF_data/2022/1020/"
    )
    path = fpath + fname
    hdul = fits.open(path)
    # print(path)
    file1_array = hdul[quad].data
    norm = (file1_array - np.min(file1_array)) / (
        np.max(file1_array) - np.min(file1_array)
    )  # norm
    image = file1_array
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["image.interpolation"] = "nearest"
    plt.rcParams["image.cmap"] = "gist_gray"
    # plt.imshow(norm)
    # plt.xlim(1070,1170)
    # plt.ylim(100,120)

    column = int(col)

    # note: the scatter plots are shifted in the positive x direction by wid,
    # so that they line up with the images

    gs_kw = dict(width_ratios=[2, 2], height_ratios=[1, 2])
    fig, axd = plt.subplot_mosaic(
        [["upper left", "upper mid"], ["lower left", "lower mid"]],
        gridspec_kw=gs_kw,
        figsize=(7.5, 3.5),
        layout="constrained",
    )
    
    background = get_image_background(image)

    imagenobackground = image - np.full(np.shape(image), background)
    
    if csv==False:
        tot, maxcols = get_tot(df, index, half_aperture, tracking)
    else:
        tot, maxcols = get_tot_csv(df, index, lc_filename=lc_filename, half_aperture=half_aperture, lcfpath = lcfpath)

    if plot_tracking == True:
        axd["upper left"].plot(maxcols + half_aperture + 1, color="yellow")
        axd["upper left"].plot(maxcols - half_aperture, color="yellow")
        axd["upper mid"].plot(maxcols + half_aperture + 1, color="yellow")
        axd["upper mid"].plot(maxcols - half_aperture, color="yellow")
    if ylims != 0:
        lower = ylims[0]
        upper = ylims[1]
        axd["lower left"].set_ylim(lower, upper)  # (7850,14000)
        axd["lower mid"].set_ylim(lower, upper)

    sqrt = np.sqrt(tot / 6.2 + 8.5**2)  # 6.2 electrons per photon, read noise is 8.5
    # fluxerr = np.divide(sqrt, len(column_indices))
    fluxerr = sqrt
    streakavg = tot  # i don't use this any more

    stderr = np.zeros(len(tot))
    for i in range(len(stderr)):
        if i > 60 and i < (len(stderr) - 60):
            arr = np.concatenate((tot[i - 60 : i - 10], tot[i + 10 : i + 60]))
            stderr[i] = np.std(arr)

    axd["lower left"].plot(
        np.arange(len(streakavg))[row - 9 : row + 9],
        tot[row - 9 : row + 9],
        color="white",
        linestyle="none",
        marker=".",
        markersize=7,
        markeredgecolor="black",
        zorder=10,
    )

    axd["lower mid"].plot(
        np.arange(len(streakavg))[row - 9 : row + 9],
        tot[row - 9 : row + 9],
        color="white",
        linestyle="none",
        marker=".",
        markersize=7,
        markeredgecolor="black",
        zorder=10,
    )

    axd["lower left"].errorbar(
        np.arange(len(streakavg)),
        tot,
        yerr=fluxerr,
        color="black",
        ecolor="black",
        elinewidth=1,
        linestyle="none",
        marker=".",
        markersize=1,
        alpha=0.4,
        label="Shot Noise",
    )

    axd["lower left"].errorbar(
        np.arange(len(streakavg)),
        tot,
        yerr=stderr,
        color="black",
        ecolor="SteelBlue",
        elinewidth=1,
        linestyle="none",
        marker=".",
        markersize=1,
        alpha=0.2,
        label=r"$\sigma_{Flux}$",
    )

    axd["lower mid"].errorbar(
        np.arange(len(streakavg)),
        tot,
        yerr=fluxerr,
        color="black",
        ecolor="black",
        elinewidth=1,
        linestyle="none",
        marker=".",
        markersize=1,
        alpha=0.4,
    )

    axd["lower mid"].errorbar(
        np.arange(len(streakavg)),
        tot,
        yerr=stderr,
        color="black",
        ecolor="SteelBlue",
        elinewidth=1,
        linestyle="none",
        marker=".",
        markersize=1,
        alpha=0.2,
    )

    #     axd['lower left'].fill_between(np.arange(len(streakavg)), tot+fluxerr/2, tot-fluxerr/2,
    #                                    alpha=.5, label="Poisson error")
    #                 # color="black", ecolor='blue', elinewidth=1,linestyle='none',
    #                 # marker=".", markersize=1, alpha=.5, label="Poisson error")

    #     axd['lower left'].fill_between(np.arange(len(streakavg)), tot+stderr/2, tot-stderr/2,
    #                                    alpha=.3, label="Stdv error")
    # axd['lower left'].fill_between(np.arange(len(streakavg)), tot, yerr=stderr,
    #                 color="black", ecolor='green', elinewidth=1,linestyle='none',
    #                 marker=".", markersize=1, alpha=.5, label="Stdv error")

    axd["upper left"].imshow(np.sqrt(image).T, aspect=20, cmap="gist_gray")
    axd["upper left"].set_ylim(column - 20, column + 20)

    axd["upper left"].set_xticklabels([])
    axd["upper left"].set_yticklabels([])
    axd["upper left"].set_yticks([])
    axd["upper left"].set_xticks([])

    # axd['lower left'].set_xti3cks(np.linspace(0, len(image[:, column]), 6))

    # axd['lower left'].set_xticklabels(np.round(np.linspace(0, len(image[:, column])*10/6000, 6*10/6000)), fontsize=12)
    axd["lower left"].set_xticks(np.linspace(0, 3080 * 10 / 11, 10))
    axd["lower left"].set_xticklabels(
        np.round(np.linspace(0, 9, 10, dtype="int")), fontsize=13
    )

    axd["lower left"].set_xlim(0, len(image[:, column]))
    axd["lower left"].set_xlabel("Time (seconds)", fontsize=12)
    axd["lower left"].set_ylabel("Flux (counts)", fontsize=11)
    axd["upper left"].set_ylabel("Image", fontsize=11)

    axd["upper mid"].imshow(np.sqrt(image).T, aspect=0.3, cmap="gist_gray")
    axd["upper mid"].set_xlim(row - 20, row + 20)
    axd["upper mid"].set_ylim(column - 20, column + 20)

    # axd['lower mid'].set_xlim(row-30,row+30)
    axd["lower mid"].set_xlim(row - 20, row + 20)
    # axd['lower mid'].set_xticks(np.linspace(row-20,row+20, 3))
    axd["lower mid"].set_xticks(
        np.linspace(row - 16, row + 16, 3),
        labels=np.linspace((row - 16) * 10 / 3, (row + 16) * 10 / 3, 3, dtype="int"),
        fontsize=13,
    )

    axd["upper mid"].set_axis_off()
    axd["lower mid"].set_yticks([])
    # axd['lower mid'].set_xticks(np.linspace(row-30,row+30, 3))
    axd["lower mid"].set_xlabel("Time (milliseconds)", fontsize=12)

    # axd['lower left'].legend([" ","Poisson err.", "Stdv err."])
    axd["lower left"].legend()
    if title == None:
        fig.suptitle("Possible Transient # " + str(index), fontsize=24)
    else:
        fig.suptitle(title, fontsize=24)

    plt.show()
    plt.close()


def get_milliseconds_from_khz(freq):
    """
    put in frequency in khz, get:
    mspercycle, secpercycle, pixpercycle
    """
    # we have cycles/millisecond
    mspercycle = 1 / freqtot
    secpercycle = mspercycle / 1000
    pixpercycle = mspercycle * 3.08 / 10
    return mspercycle, secpercycle, pixpercycle


def add_snr(df, direcname, half_aperture, tracking):
    """
    returns the dataframe with a new column of stdv ratios
    see get_sobel_ratio(stamp) and get_stdv_ratio
    """
    ratios = np.zeros(len(df))
    for index in df.index:
        # print(index)
        tot, maxcols = get_tot(df, index, half_aperture, tracking)
        stdv = get_stdvs(tot, do_all=False, df=df, index=index)
        avg = np.average(tot)
        row = int(df.loc[index]["row_from_nn"]) + 10
        signal = tot[row] - avg
        ratios[index] = signal / stdv
    df["snr"] = ratios
    return df


def get_tot_streak(image, col, tracking=True, half_aperture=2):
    """ """
    # row = int(df.iloc[index]["row_from_nn"])+10 #we add 10 here bc that's the cnn offset of returned values
    # col = int(df.iloc[index]["col_from_nn"])+10
    # col = int(peaks[peakindex])
    row = 1500  # for calculating avg
    background = get_image_background(image)
    imagenobackground = image - np.full(np.shape(image), background)
    if tracking == False:
        zeropoint = col - 7
        averages = np.average(
            imagenobackground[row - 50 : row + 50, zeropoint : zeropoint + 14], axis=0
        )
        maxind = np.argmax(averages)
        tot2 = np.zeros(imagenobackground.shape[0])
        for r in range(
            0, imagenobackground.shape[0]
        ):  # this should be each row, probably
            maxcol = zeropoint + maxind
            combined = imagenobackground[r, maxcol]
            for i in range(1, half_aperture + 1):
                combined += imagenobackground[r, maxcol - i]
                # combined+=imagenobackground[r, maxcol+i]
            tot2[r] = combined
        return tot2

    # so we need to track the whole streak
    # for each row, need to choose a column index
    tot = np.zeros(imagenobackground.shape[0])
    maxcols = np.zeros(imagenobackground.shape[0])
    zeropoint = col - 10
    for r in range(0, imagenobackground.shape[0]):  # this should be each row, probably
        maxcol = np.argmax(imagenobackground[r, zeropoint : zeropoint + 20]) + zeropoint
        maxcols[r] = maxcol
        combined = imagenobackground[r, maxcol]
        for i in range(1, half_aperture + 1):
            combined += imagenobackground[r, maxcol - i]
            combined += imagenobackground[r, maxcol + i]
        # combined = imagenobackground[r, maxcol-2] + imagenobackground[r, maxcol-1] + imagenobackground[r, maxcol] + imagenobackground[r, maxcol+1] + imagenobackground[r, maxcol+2]
        # print(combined)
        tot[r] = combined

    return tot, maxcols






def get_initial_df_for_lc(lc_filename, tolerance=10):
    '''
    used for plot_lc_from_csv_with_transient_df
    lc_filename: filename of saved light curve
    tolerance: how far from the streak the transient has to be
    in order for it to count as on the streak
    default: ten pixels
    returns a dataframe
    '''
    df_path = (
    '/lustre/lrspec/ZTF_data/science_extractions/1020/')
    filename_start = lc_filename[11:50]
    quad = lc_filename[51]
    df_filename = filename_start+'.fits.fz'+quad+'alllist'
    print(df_filename)
    df = pd.read_csv(df_path+df_filename)
    #i think we're gonna ignore the "both" for now
    df = df[df['type']!='both']
    #so do we think that row is x and column is y? seems likely
    lc_x_coord = int(lc_filename[53:57])
    #we want col from cnn as our streak column
    detection_col = df['col_from_nn']+10
    df = df[np.abs(lc_x_coord-detection_col)<tolerance]
    return df
    
    
def get_second_df_for_lc(lc_filename, tolerance=10):
    '''
    used for plot_lc_from_csv_with_transient_df
    lc_filename: filename of saved light curve
    tolerance: how far from the streak the transient has to be
    in order for it to count as on the streak
    default: ten pixels
    returns a dataframe
    '''
    df_path = (
    '/lustre/lrspec/ZTF_data/science_extractions/1020/')
    filename_start = lc_filename[11:50]
    quad = lc_filename[51]
    df_filename = filename_start+'.fits.fz'+quad+'alllist'+ "_reduced"    
    print(df_filename)
    df = pd.read_csv(df_path+df_filename)
    #i think we're gonna ignore the "both" for now
    df = df[df['type']!='both']
    #so do we think that row is x and column is y? seems likely
    lc_x_coord = int(lc_filename[53:57])
    #we want col from cnn as our streak column
    detection_col = df['col_from_nn']+10
    df = df[np.abs(lc_x_coord-detection_col)<tolerance]
    # print(len(df))
    return df

def get_third_df_for_lc(lc_filename, tolerance=10):
    '''
    used for plot_lc_from_csv_with_transient_df
    lc_filename: filename of saved light curve
    tolerance: how far from the streak the transient has to be
    in order for it to count as on the streak
    default: ten pixels
    returns a dataframe
    '''
    df_path = (
    '/lustre/lrspec/ZTF_data/science_extractions/1020/')
    filename_start = lc_filename[11:50]
    quad = lc_filename[51]
    df_filename = filename_start+'.fits.fz'+quad+'alllist'+ "_reduced" +"_sobelratio"+"_snr" 
    print(df_filename)
    df = pd.read_csv(df_path+df_filename)
    # return df
    #i think we're gonna ignore the "both" for now
    df = df[df['type']!='both']
    #so do we think that row is x and column is y? seems likely
    lc_x_coord = int(lc_filename[53:57])
    #we want col from cnn as our streak column
    detection_col = df['col_from_nn']+10
    df = df[np.abs(lc_x_coord-detection_col)<tolerance]
    return df

def get_last_df_for_lc(lc_filename):
    '''
    used for plot_lc_from_csv_with_transient_df
    lc_filename: filename of saved light curve
    tolerance: how far from the streak the transient has to be
    in order for it to count as on the streak
    default: ten pixels
    returns a dataframe
    '''
    df_last = pd.read_csv("/home/3155/best_cands_nosim")
    filename_start = lc_filename[11:50]+'.'
    df_last_ourfile = df_last[df_last['filename']==filename_start]

    quad = lc_filename[51]
    
    df_last_ourquad = df_last_ourfile[df_last_ourfile['quadrant']==int(quad)]
    lc_x_coord = int(lc_filename[53:57])
    df_last_ourstreak = df_last_ourquad[np.abs(df_last_ourquad['closest_x']-lc_x_coord)<1]
    return df_last_ourstreak


def get_withsim_df_for_lc(lc_filename):
    '''
    used for plot_lc_from_csv_with_transient_df
    lc_filename: filename of saved light curve
    tolerance: how far from the streak the transient has to be
    in order for it to count as on the streak
    default: ten pixels
    returns a dataframe
    '''
    df_last = pd.read_csv("/home/3155/best_cands")
    filename_start = lc_filename[11:50]+'.'
    df_last_ourfile = df_last[df_last['filename']==filename_start]

    quad = lc_filename[51]
    
    df_last_ourquad = df_last_ourfile[df_last_ourfile['quadrant']==int(quad)]
    lc_x_coord = int(lc_filename[53:57])
    df_last_ourstreak = df_last_ourquad[np.abs(df_last_ourquad['closest_x']-lc_x_coord)<1]
    return df_last_ourstreak


def plot_lc_from_csv_with_transient_df(lc_filename, df, rowindex, hap=2):
    '''
    plots light curve with transient candidates, color coded to
    which part of the pipeline they were removed on
    give it lc_filename OR df and rowindex (make lc_filename=None)
    hap: half aperture for light curve, allowable from 2 to 5
    '''
    if lc_filename==None:
        lightcurve_path = '/lustre/lrspec/ZTF_data/lightcurves/1020/'
        df_row = df.loc[rowindex]
        lc_filename = 'lightcurve_'+df_row['filename'][:-1]+'_'+str(df_row['quadrant'])+'_'+str(int(df_row['closest_x']))+'_'+str(int(df_row['closest_y']))+'.csv'
    
    lightcurve_path = '/lustre/lrspec/ZTF_data/lightcurves/1020/'
    lc_df = pd.read_csv(lightcurve_path+lc_filename)
    tot = lc_df['flux_hap_'+str(hap)]
    # plt.plot(tot, zorder=10)
    fluxerr = np.sqrt(tot / 6.2 + 8.5**2)  # 6.2 electrons per photon, read noise is 8.5
    plt.rcParams["figure.figsize"] = (10, 5)
    fig, ax = plt.subplots()
    ax.errorbar(
        np.arange(len(tot)),
        tot,
        yerr=fluxerr,
        color="black",
        ecolor="black",
        elinewidth=1,
        linestyle="none",
        marker=".",
        markersize=1,
        alpha=0.4,
        label="Shot Noise",
        zorder=10
    )
    transients_df = get_initial_df_for_lc(lc_filename, tolerance=10)
    for i, row in transients_df.iterrows():
        # print(i)
        row_num = (int(row['row_from_nn'])+10)
        if i==transients_df.index[0]:
            plt.axvline(x=row_num,c='mistyrose', label = "CNN marked as a cosmic ray")
        else:
            plt.axvline(x=row_num,c='mistyrose')
        ax.plot(
        np.arange(len(tot))[row_num - 9 : row_num + 9],
        tot[row_num - 9 : row_num + 9],
        color="white",
        linestyle="none",
        marker=".",
        markersize=7,
        markeredgecolor="black",
        zorder=10,
    )
#     #three things happen:
#     # Reduce based on type != 'cr'
    df_nocr = transients_df[transients_df["type"] != "cr"]
    for i, row in df_nocr.iterrows():
        # print(i)
        row_num = (int(row['row_from_nn'])+10)
        if i==df_nocr.index[0]:
            plt.axvline(x=row_num,c='thistle', label = "eliminated: too close to edges")
        else:
            plt.axvline(x=row_num,c='thistle')
    df_reduced_edges= reduce_edges(df_nocr, 15)
    for i, row in df_reduced_edges.iterrows():
        # print(i)
        row_num = (int(row['row_from_nn'])+10)
        if i==df_reduced_edges.index[0]:
            plt.axvline(x=row_num,c='paleturquoise', label = "eliminated: total size < 4 pixels")
        else:
            plt.axvline(x=row_num,c='paleturquoise')
    df_reduced_edges_size4 = reduce_size(df_reduced_edges, 4)
    for i, row in df_reduced_edges_size4.iterrows():
    # print(i)
        row_num = (int(row['row_from_nn'])+10)
        if i==df_reduced_edges_size4.index[0]:
            plt.axvline(x=row_num,c='lemonchiffon', label = "eliminated: not sufficiently bright")
        else:
            plt.axvline(x=row_num,c='lemonchiffon')
#     #reduce_max_3 got saved as a new df
    
    df_second = get_second_df_for_lc(lc_filename, tolerance=10)
    for i, row2 in df_second.iterrows():
        # print(i)
        row_num = (int(row2['row_from_nn'])+10)
        if i==df_second.index[0]:
            plt.axvline(x=row_num,c='lightsteelblue', label = "eliminated: sobel ratio < 5 (cosmic ray)")
        else:
            plt.axvline(x=row_num,c='lightsteelblue')  
    df_third = get_third_df_for_lc(lc_filename, tolerance=10)
    for i, row in df_third.iterrows():
    # print(i)
        row_num = (int(row['row_from_nn'])+10)
        if i==df_third.index[0]:
            plt.axvline(x=row_num,c='pink', label = "eliminated: local snr<1")
        else:
            plt.axvline(x=row_num,c='pink')
        if np.abs(row['snr_center'])>25:
            plt.axvline(x=row_num,c='orange', label = "eliminated: moving object (snr>25)")
            
    #i definitely did not include anything with SNR>25 at this point but i dont think i saved that
    #the only other strict cut was local snr>1
    #if i add the simultaneity criterion then i suppose i should put that here as well
    
    df_last = get_last_df_for_lc(lc_filename)
    for i, row in df_last.iterrows():
    # print(i)
        row_num = (int(row['row_from_nn'])+10)
        if i==df_last.index[0]:
            plt.axvline(x=row_num,c='palegreen', label = "remains")
        else:
            plt.axvline(x=row_num,c='palegreen')
    
    ax.set_title("Lightcurve with Transient Candidate Cuts",fontsize=20)
    ax.set_ylabel("Flux (counts)", fontsize=11)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_xticks(np.linspace(0, 3080 * 10 / 11, 10))
    ax.set_xticklabels(
        np.round(np.linspace(0, 9, 10, dtype="int")), fontsize=13
    )
    ax.legend(bbox_to_anchor=(1.1, 1.05),fontsize=18)
    fig.show()
    # return lc_df
    
    
    
    


def plot_lc_from_csv_with_transient_df_plotly(lc_filename, df=None, rowindex=None, hap=2, old_lines=True):
    '''
    Creates an interactive Plotly version of my light curve plot with transient candidates
    
    Parameters:
    lc_filename: path to lightcurve CSV file - probably best to put this as none and use df and rowindex
    df: dataframe 
    rowindex: row index 
    hap: half aperture for light curve, allowable from 2 to 5
    '''
    
    # Handle filename logic 
    if lc_filename == None:
        lightcurve_path = '/lustre/lrspec/ZTF_data/lightcurves/1020/'
        df_row = df.loc[rowindex]
        lc_filename = 'lightcurve_'+df_row['filename'][:-1]+'_'+str(df_row['quadrant'])+'_'+f'{int(df_row["closest_x"]):04}'+'_'+f'{int(df_row["closest_y"]):04}'+'.csv'
    

    lightcurve_path = '/lustre/lrspec/ZTF_data/lightcurves/1020/'  
    lc_df = pd.read_csv(lightcurve_path + lc_filename)
    
    # Calculate flux and error
    tot = lc_df['flux_hap_'+str(hap)]
    fluxerr = np.sqrt(tot / 6.2 + 8.5**2)  # 6.2 electrons per photon, read noise is 8.5
    time_points = np.arange(len(tot))
    
    # Create the main plot
    fig = go.Figure()
    

    
    # Define colors for different pipeline stages
    colors = {
        'cnn_cosmic_ray': 'rgba(255, 228, 225, 1)',  # mistyrose
        'edges': 'rgba(216, 191, 216, 1)',            # thistle
        'size': 'rgba(175, 238, 238, 1)',             # paleturquoise
        'brightness': 'rgba(255, 250, 205, 1)',       # lemonchiffon
        'sobel': 'rgba(176, 196, 222, 1)',            # lightsteelblue
        'snr': 'rgba(255, 192, 203, 1)',              # pink
        'moving': 'rgba(255, 165, 0, 1)',             # orange
        'remains': 'rgba(152, 251, 152, 1)'           # palegreen
    }
    
    def add_transient_lines(df_transients, color, label, fig, highlight=False):
        if len(df_transients) == 0:
            return
        
        # Get y-axis range for vertical lines
        y_min, y_max = min(tot), max(tot)
        y_range = y_max - y_min
        y_min_extended = y_min - 0.1 * y_range
        y_max_extended = y_max + 0.1 * y_range
        
        for i, (idx, row) in enumerate(df_transients.iterrows()):
            row_num = int(row['row_from_nn']) + 10
            
            # Add vertical line as a shape (appears behind data)
            fig.add_shape(
                type="line",
                x0=row_num, x1=row_num,
                y0=y_min_extended, y1=y_max_extended,
                line=dict(color=color, width=2),
                opacity=1,
                layer="below"  # This puts it behind the data
            )
            
            # Add invisible trace for legend (only for first occurrence)
            if i == 0:
                fig.add_trace(go.Scatter(
                    x=[row_num], y=[y_max_extended],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=label,
                    showlegend=True,
                    # opacity=0  # Make the trace itself invisible but keep legend normal
                ))
            
            # Highlight the data points around the transient
            if highlight==True:
                if row_num - 9 >= 0 and row_num + 9 < len(tot):
                    highlight_x = time_points[row_num - 9 : row_num + 9]
                    highlight_y = tot[row_num - 9 : row_num + 9]
                    
                    fig.add_trace(go.Scatter(
                        x=highlight_x,
                        y=highlight_y,
                        mode='markers',
                        marker=dict(size=8, color='white', line=dict(color='black', width=1)),
                        showlegend=False,
                        hovertemplate='<b>Transient Candidate</b><br><b>Time:</b> %{x}<br><b>Flux:</b> %{y:.2f}<extra></extra>'
                    ))
    

    
    # Example of how to structure the data loading:
    # Replace these with your actual data loading functions
    try:
        # These would be your actual function calls:
        if old_lines==True:
            transients_df = get_initial_df_for_lc(lc_filename, tolerance=10)
            df_nocr = transients_df[transients_df["type"] != "cr"]
            df_reduced_edges = reduce_edges(df_nocr, 15)
            df_reduced_edges_size4 = reduce_size(df_reduced_edges, 4)
            df_second = get_second_df_for_lc(lc_filename, tolerance=10)
            df_third = get_third_df_for_lc(lc_filename, tolerance=10)
            df_last = get_last_df_for_lc(lc_filename)
            
            df_withsim = get_withsim_df_for_lc(lc_filename)
        

        
            # Add the transient markers (you'll replace this with your actual data)
            add_transient_lines(transients_df, colors['cnn_cosmic_ray'], "CNN marked as cosmic ray", fig)
            add_transient_lines(df_nocr, colors['edges'], "eliminated: too close to edges", fig)
            add_transient_lines(df_reduced_edges, colors['size'], "eliminated: total size < 4 pixels", fig)
            
            add_transient_lines(df_second, colors['sobel'], "eliminated: sobel ratio < 5 (cosmic ray)", fig)
            add_transient_lines(df_third, colors['snr'], "eliminated: local snr<1", fig)
            add_transient_lines(df_withsim, colors['brightness'], "eliminated: has simultaneous events", fig)

    except Exception as e:
        print(f"Note: Could not load transient data - {e}")
        print("You'll need to adapt the data loading part for your specific setup")
 
        # Add main light curve with error bars
    fig.add_trace(go.Scatter(
        x=time_points,
        y=tot,
        error_y=dict(type='data', color='rgba(0,0,0,0.5)',array=fluxerr, visible=True,width=0),
        mode='markers',
        marker=dict(size=3, color='black', opacity=0.3),
        name='Shot Noise',
        hovertemplate='<b>Time:</b> %{x}<br><b>Flux:</b> %{y:.2f}<br><b>Error:</b> %{error_y.array:.2f}<extra></extra>'
    ))
    
    #maybe this will show up on top
    if old_lines==True:
        add_transient_lines(df_last, colors['remains'], "Remaining transient candidate", fig, highlight=True)
    else:
        if int(df_row['row_from_nn']) + 10 - 9 >= 0 and int(df_row['row_from_nn']) + 10 + 9 < len(tot):
                    highlight_x = time_points[int(df_row['row_from_nn']) + 10 - 9 : int(df_row['row_from_nn']) + 10 + 9]
                    highlight_y = tot[int(df_row['row_from_nn']) + 10 - 9 : int(df_row['row_from_nn']) + 10 + 9]
                    
                    fig.add_trace(go.Scatter(
                        x=highlight_x,
                        y=highlight_y,
                        mode='markers',
                        marker=dict(size=8, color='white', line=dict(color='black', width=1)),
                        showlegend=False,
                        hovertemplate='<b>Transient Candidate</b><br><b>Time:</b> %{x}<br><b>Flux:</b> %{y:.2f}<extra></extra>'
                    ))

    # Update layout to match your original styling
    fig.update_layout(
        title=dict(
            text=f"Lightcurve #{rowindex} with Half-Aperture {hap} with Transient Candidate Cuts",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Time (seconds)",
            # titlefont=dict(size=12),
            tickfont=dict(size=13),
            tickmode='array',
            tickvals=np.linspace(0, 3080 * 10 / 11, 10),
            ticktext=[str(int(x)) for x in np.linspace(0, 9, 10)]
        ),
        yaxis=dict(
            title="Flux (counts)",
            # titlefont=dict(size=11)
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left", 
            x=1.02,
            font=dict(size=14)
        ),
        template='plotly_white',
        width=1300,
        height=500
    )
    

    
    return fig


def plot_transient_streak_plotly(
    df,
    index,
    direcname="1020",
    half_aperture=2,
    tracking=True,
    title=None,
    plot_tracking=False,
):
    """
    Creates a Plotly version of the upper left image plot from your matplotlib function
    
    Parameters:
    df: dataframe with image information
    index: row index to plot
    direcname: directory name for images
    half_aperture: aperture size parameter
    tracking: whether to track brightest pixel
    title: custom title for the plot
    plot_tracking: whether to show tracking lines
    
    Returns:
    Plotly figure object
    """
    
    # Extract data from dataframe (same as original)
    wid = 10
    fname = df.loc[index]["filename"] + "fits.fz"
    row = int(df.loc[index]["row_from_nn"]) + 10
    col = int(df.loc[index]["col_from_nn"]) + 10
    quad = int(df.loc[index]["quadrant"])
    column = int(col)
    
    # Load the image (same as original)
    fpath = (
        "/lustre/lrspec/ZTF_data/2022/1020/"
    )
    path = fpath + fname
    
    try:
        hdul = fits.open(path)
        file1_array = hdul[quad].data
        hdul.close()
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Process image (same as original)
    image = file1_array
    sqrt_image = np.sqrt(image).T  # Apply sqrt and transpose like original
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add the image as a heatmap
    fig.add_trace(go.Heatmap(
        z=sqrt_image,
        colorscale='gray',
        showscale=True,
        colorbar=dict(
            title="Counts",
            # titleside="right"
        ),
        hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Intensity:</b> %{z:.2f}<extra></extra>'
    ))
    
    # Add square outline showing the "upper mid" zoomed region
    # This corresponds to the region shown in your original upper mid plot
    zoom_x_min = row - 20
    zoom_x_max = row + 20
    zoom_y_min = column - 20
    zoom_y_max = column + 20
    



    
    # Set the layout to match the original
    
    
    fig.update_layout(
        title=dict(
            text=title if title else f"Close View - Transient #{index}",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[zoom_x_min, zoom_x_max]
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[column - 10, column + 10],  # Match original ylim
            # scaleanchor="x",
            # scaleratio=20  # Match original aspect=20
        ),
        plot_bgcolor='black',
        paper_bgcolor='white',
        width=1000,
        height=300,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig


def plot_whole_streak_plotly(
    df,
    index,
    direcname="1020",
    half_aperture=2,
    tracking=True,
    title=None,
    plot_tracking=False,
):
    """
    Creates a Plotly version of the upper left image plot from your matplotlib function
    
    Parameters:
    df: dataframe with image information
    index: row index to plot
    direcname: directory name for images
    half_aperture: aperture size parameter
    tracking: whether to track brightest pixel
    title: custom title for the plot
    plot_tracking: whether to show tracking lines
    
    Returns:
    Plotly figure object
    """
    
    # Extract data from dataframe (same as original)
    wid = 10
    fname = df.loc[index]["filename"] + "fits.fz"
    row = int(df.loc[index]["row_from_nn"]) + 10
    col = int(df.loc[index]["col_from_nn"]) + 10
    quad = int(df.loc[index]["quadrant"])
    column = int(col)
    
    # Load the image (same as original)
    fpath = (
        "/lustre/lrspec/ZTF_data/2022/1020/"
    )
    path = fpath + fname
    
    try:
        hdul = fits.open(path)
        file1_array = hdul[quad].data
        hdul.close()
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Process image (same as original)
    image = file1_array
    sqrt_image = np.sqrt(image).T  # Apply sqrt and transpose like original
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add the image as a heatmap
    fig.add_trace(go.Heatmap(
        z=sqrt_image,
        colorscale='gray',
        showscale=True,
        colorbar=dict(
            title="Counts",
            # titleside="right"
        ),
        hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Intensity:</b> %{z:.2f}<extra></extra>'
    ))
    
    # Add square outline showing the "upper mid" zoomed region
    # This corresponds to the region shown in your original upper mid plot
    zoom_x_min = row - 20
    zoom_x_max = row + 20
    zoom_y_min = column - 20
    zoom_y_max = column + 20
    
    # Add rectangle outline
    fig.add_shape(
        type="rect",
        x0=zoom_x_min, x1=zoom_x_max,
        y0=zoom_y_min, y1=zoom_y_max,
        line=dict(color="red", width=2),
        fillcolor="rgba(0,0,0,0)",  # Transparent fill
        layer="above"
    )
    
    # Add a small annotation to label the zoom region
    fig.add_annotation(
        x=zoom_x_max,
        y=zoom_y_max,
        text="Transient area",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        font=dict(color="red", size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
        borderwidth=1
    )
    
    # Add tracking lines if requested
    if plot_tracking:
        # You'd need to implement get_tot function or pass tracking data
        # For now, this is a placeholder for the tracking lines
        try:
            # This would need your get_tot function
            # tot, maxcols = get_tot(df, index, half_aperture, tracking)
            # 
            # # Add tracking lines
            # fig.add_trace(go.Scatter(
            #     x=list(range(len(maxcols))),
            #     y=maxcols + half_aperture + 1,
            #     mode='lines',
            #     line=dict(color='yellow', width=2),
            #     name='Upper tracking',
            #     showlegend=False
            # ))
            # 
            # fig.add_trace(go.Scatter(
            #     x=list(range(len(maxcols))),
            #     y=maxcols - half_aperture,
            #     mode='lines',
            #     line=dict(color='yellow', width=2),
            #     name='Lower tracking',
            #     showlegend=False
            # ))
            pass
        except:
            print("Tracking lines not implemented - need get_tot function")
    
    # Set the layout to match the original
    
    
    fig.update_layout(
        title=dict(
            text=title if title else f"Whole Streak View - Transient #{index}",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, sqrt_image.shape[1]]
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[column - 20, column + 20],  # Match original ylim
            scaleanchor="x",
            scaleratio=20  # Match original aspect=20
        ),
        plot_bgcolor='black',
        paper_bgcolor='white',
        width=1300,
        height=300,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig


def plot_whole_image_plotly(
    df,
    index,
    direcname="1020",
    half_aperture=2,
    tracking=True,
    title=None,
    plot_tracking=False,
):
    """
    Creates a Plotly version of the upper left image plot from your matplotlib function
    
    Parameters:
    df: dataframe with image information
    index: row index to plot
    direcname: directory name for images
    half_aperture: aperture size parameter
    tracking: whether to track brightest pixel
    title: custom title for the plot
    plot_tracking: whether to show tracking lines
    
    Returns:
    Plotly figure object
    """
    
    # Extract data from dataframe (same as original)
    wid = 10
    fname = df.loc[index]["filename"] + "fits.fz"
    row = int(df.loc[index]["row_from_nn"]) + 10
    col = int(df.loc[index]["col_from_nn"]) + 10
    quad = int(df.loc[index]["quadrant"])
    column = int(col)
    
    # Load the image (same as original)
    fpath = (
        "/lustre/lrspec/ZTF_data/2022/1020/"
    )
    path = fpath + fname
    
    try:
        hdul = fits.open(path)
        file1_array = hdul[quad].data
        hdul.close()
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Process image (same as original)
    image = file1_array
    sqrt_image = np.sqrt(image).T  # Apply sqrt and transpose like original
    
    zmin = float(np.nanmin(sqrt_image))
    zmax = float(np.nanmax(sqrt_image))
    
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add the image as a heatmap
    fig.add_trace(go.Heatmap(
        z=sqrt_image,
        zmin=zmin,
        zmax=zmax,
        colorscale='gray',
        showscale=True,
        colorbar=dict(
            title="Counts",
            # titleside="right"
        ),
        hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Intensity:</b> %{z:.2f}<extra></extra>'
    ))
    
    # Add square outline showing the "upper mid" zoomed region
    # This corresponds to the region shown in your original upper mid plot
    zoom_x_min = row - 20
    zoom_x_max = row + 20
    zoom_y_min = column - 20
    zoom_y_max = column + 20
    
    # Add rectangle outline
    fig.add_shape(
        type="rect",
        x0=zoom_x_min, x1=zoom_x_max,
        y0=zoom_y_min, y1=zoom_y_max,
        line=dict(color="red", width=2),
        fillcolor="rgba(0,0,0,0)",  # Transparent fill
        layer="above"
    )
    
    # Add a small annotation to label the zoom region
    fig.add_annotation(
        x=zoom_x_max,
        y=zoom_y_max,
        text="Transient area",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        font=dict(color="red", size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
        borderwidth=1
    )
    
    # Add tracking lines if requested
    if plot_tracking:
        # You'd need to implement get_tot function or pass tracking data
        # For now, this is a placeholder for the tracking lines
        try:
            # This would need your get_tot function
            # tot, maxcols = get_tot(df, index, half_aperture, tracking)
            # 
            # # Add tracking lines
            # fig.add_trace(go.Scatter(
            #     x=list(range(len(maxcols))),
            #     y=maxcols + half_aperture + 1,
            #     mode='lines',
            #     line=dict(color='yellow', width=2),
            #     name='Upper tracking',
            #     showlegend=False
            # ))
            # 
            # fig.add_trace(go.Scatter(
            #     x=list(range(len(maxcols))),
            #     y=maxcols - half_aperture,
            #     mode='lines',
            #     line=dict(color='yellow', width=2),
            #     name='Lower tracking',
            #     showlegend=False
            # ))
            pass
        except:
            print("Tracking lines not implemented - need get_tot function")
    
    # Set the layout to match the original

    
    fig.update_layout(
        title=dict(
            text=title if title else f"Whole Image - Transient #{index}",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, sqrt_image.shape[1]]
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            # range=[column - 20, column + 20],  # Match original ylim
            # scaleanchor="x",
            # scaleratio=20  # Match original aspect=20
        ),
        plot_bgcolor='black',
        paper_bgcolor='white',
        width=1300,
        height=300,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig


def get_tot_csv(
    df,
    index, lc_filename=None,
    half_aperture=2, lcfpath='/lustre/lrspec/ZTF_data/lightcurves/1020/'  
):
    """
    returns tot, maxcols: an array that is the flux of the star streak
    given an aperture that is halfaperture*2+1
    and if tracking=true, tracks when the streak posittion fluctuates
    oif tracking==false, goes based on average center of streak
    put in lc_filename yourself or it will take it from the dataframe 
    (in which case df and index are ignored)
    """
    # Handle filename logic 
    if lc_filename == None:
        lightcurve_path = '/lustre/lrspec/ZTF_data/lightcurves/1020/'
        df_row = df.loc[index]
        lc_filename = 'lightcurve_'+df_row['filename'][:-1]+'_'+str(df_row['quad'])+'_'+f'{int(df_row["closest_x"]):04}'+'_'+f'{int(df_row["closest_y"]):04}'+'.csv'
    
    if lcfpath==None:
        lcfpath = '/lustre/lrspec/ZTF_data/lightcurves/1020/'  
    lc_df = pd.read_csv(lcfpath + lc_filename)
    
    # Calculate flux and error
    tot = lc_df['flux_hap_'+str(half_aperture)]
    maxcols = lc_df['max_col']

    return tot, maxcols


def get_tot_npy(
    df,
    index, posneg,
    half_aperture=2,
    tracking=True
):
    """
    FOR EFFICIENCY CALC
    returns tot, maxcols: an array that is the flux of the star streak
    given an aperture that is halfaperture*2+1
    and if tracking=true, tracks when the streak posittion fluctuates
    oif tracking==false, goes based on average center of streak
    note: uses loc instead of iloc
    """
    if posneg=="pos":
        fpath = '/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_pos/images/'

    elif posneg=="neg":
        fpath = '/lustre/lrspec/ZTF_data/efficiency_testing/nosat_eff_data/full_image_implantations_neg/images/'

    wideval = half_aperture
    wid = 10
    quad = int(df.loc[index]["quadrant"])
    fname = df.loc[index]["filename"] + '_'+str(quad)+'.npy'
    row = (
        int(df.loc[index]["row_from_nn"]) + 10
    )  # we add 10 here bc that's the cnn offset of returned values
    col = int(df.loc[index]["col_from_nn"]) + 10
    column = int(col)

    path = fpath + fname
    # print(path)
    file1_array = np.load(path)
    # norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array)) #norm
    image = file1_array

    background = get_image_background(image)

    imagenobackground = image - np.full(np.shape(image), background)

    if tracking == False:
        zeropoint = col - 7
        averages = np.average(
            imagenobackground[row - 10 : row + 10, zeropoint : zeropoint + 14], axis=0
        )
        maxind = np.argmax(averages)
        tot2 = np.zeros(imagenobackground.shape[0])
        for r in range(
            0, imagenobackground.shape[0]
        ):  # this should be each row, probably
            maxcol = zeropoint + maxind
            combined = imagenobackground[r, maxcol]
            for i in range(1, half_aperture + 1):
                combined += imagenobackground[r, maxcol - i]
                combined += imagenobackground[r, maxcol + i]
            tot2[r] = combined
        return tot2, np.full(len(tot2), maxcol)

    # so we need to track the whole streak
    # for each row, need to choose a column index
    tot = np.zeros(imagenobackground.shape[0])
    maxcols = np.zeros(imagenobackground.shape[0])
    zeropoint = col - 10
    for r in range(0, imagenobackground.shape[0]):  # this should be each row, probably
        maxcol = np.argmax(imagenobackground[r, zeropoint : zeropoint + 20]) + zeropoint
        maxcols[r] = maxcol
        combined = imagenobackground[r, maxcol]
        for i in range(1, half_aperture + 1):
            combined += imagenobackground[r, maxcol - i]
            combined += imagenobackground[r, maxcol + i]
        # combined = imagenobackground[r, maxcol-2] + imagenobackground[r, maxcol-1] + imagenobackground[r, maxcol] + imagenobackground[r, maxcol+1] + imagenobackground[r, maxcol+2]
        # print(combined)
        tot[r] = combined

    return tot, maxcols



def add_sobel_ratio_fast(df, central_row='central_point_brightening', direcname='1020', size=5):
    """
    returns the dataframe with a new column of sobel ratios
    see get_sobel_ratio(stamp)
    """
    sobels = {}
    # sobels_x = {}
    # sobels_y = {}
    # i=0
    for filename, group in df.groupby('filename'):     
        fpath = (
            "/lustre/lrspec/ZTF_data/2022/"
            + direcname
            + "/"
        )
        path = fpath + filename+'fits.fz'
        with fits.open(path) as hdul:
            for idx in group.index:
                # quad = int(df.loc[idx]["quad"])
                
                stamp, bg = get_stamp_from_hdul(hdul, idx, direcname, df, norm=False, central_row=central_row, size=size)
                sobel, x, y = get_sobel_ratio(stamp-bg)
                # print(sobel)
                # print(np.shape(x))
                # print(np.shape(y))
                sobels[idx] = sobel
                # sobels_x[idx] = x
                # sobels_y[idx] = y
                # i+=1
    df["sobel_ratio"+central_row+str(size)] = df.index.map(sobels)
    # df["sobels_x"+central_row+str(size)] = df.index.map(sobels_x)
    # df["sobels_y"+central_row+str(size)] = df.index.map(sobels_y)
    return df



def get_stamp_from_hdul(hdul, index, direcname, df, norm=True, quad=None, central_row="row_from_nn", size=10):
    """
    returns a 20x20 numpy array of brightnesses centered around the detection
    located at index of dataframe df
    note that the filepath to the science images is hardcoded currently
    norm=True has the image normalized (not the stamp, the entire image)
    norm=False has it in terms of counts
    """
    # fname = df.loc[index]["filename"] + "fits.fz"
    col = int(df.loc[index]["col_from_nn"]) + 10
    row = int(df.loc[index][central_row]) + 10
    if central_row=="row_from_nn":
        row = int(df.loc[index][central_row]) + 10
    else:
        row = int(df.loc[index][central_row]) #image not nn coordiantes
    if quad==None:
        quad = int(df.loc[index]["quad"])
    file1_array = hdul[quad].data
    if norm:
        normalized = (file1_array - np.min(file1_array)) / (
            np.max(file1_array) - np.min(file1_array)
        )  # norm
        stamp = normalized[row - size : row + size, col - size : col + size]
        # plt.imshow(stamp)
        # plt.colorbar()
        background = get_image_background(normalized)
        return stamp, background
        
    stamp = file1_array[row - size : row + size, col - size : col + size]
    # plt.imshow(stamp)
    # plt.colorbar()
    background = get_image_background(file1_array)
    return stamp, background

def get_stamp_from_df_eff(index, df, posneg=None, norm=False, quad=None):
    """
    returns a 20x20 numpy array of brightnesses centered around the detection
    located at index of dataframe df
    norm=True has the image normalized (not the stamp, the entire image)
    norm=False has it in terms of counts
    """
    
    row = int(df.loc[index]["row_from_nn"]) + 10
    col = int(df.loc[index]["col_from_nn"]) + 10
    posneg = df.loc[index]["type"]
    
    if quad==None:
        quad = int(df.loc[index]["quadrant"])
        
    if posneg=="pos":
        extension = "_1_implanted_pos.npy"
        fpath = '/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_pos/images/'

    elif posneg=="neg":
        extension = "_1_implanted_neg.npy"
        fpath = '/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_neg/images/'
        
    fname = df.loc[index]["filename"] + extension
    
    path = fpath + fname
    # hdul = fits.open(path)
    file1_array = np.load(path)
    if norm:
        normalized = (file1_array - np.min(file1_array)) / (
            np.max(file1_array) - np.min(file1_array)
        )  # norm
        stamp = normalized[row - 10 : row + 10, col - 10 : col + 10]
        # plt.imshow(stamp)
        # plt.colorbar()
        return stamp
        background = get_image_background(normalized)
    stamp = file1_array[row - 10 : row + 10, col - 10 : col + 10]
    # plt.imshow(stamp)
    # plt.colorbar()
    background = get_image_background(file1_array)
    # plt.imshow(stamp)
    return stamp, background



def add_stamps(df, direcname='1020'):
    """
    returns the dataframe with new columns of stamp, norm stamp

    """
    stamps = {}
    norm_stamps = {}
    # i=0
    for filename, group in df.groupby('filename'):     
        fpath = (
            "/lustre/lrspec/ZTF_data/2022/1020/"
        )
        path = fpath + filename+'fits.fz'
        # print(path)
        with fits.open(path) as hdul:
            for idx in group.index:
                # quad = int(df.loc[idx]["quad"])
                
                stamp = get_stamp_from_hdul(hdul, idx, direcname, df, norm=False)
                normstamp = get_stamp_from_hdul(hdul, idx, direcname, df, norm=True)
                # print(sobel)
                # print(np.shape(x))
                # print(np.shape(y))
                stamps[idx] = stamp
                norm_stamps[idx] = normstamp
                # bg = get_image_background(normalized)
                # i+=1
                del stamp
                del normstamp
                gc.collect()
    df["stamp"] = df.index.map(stamps)
    df["norm_stamp"] = df.index.map(norm_stamps)
    return df

def add_stamps2(df, direcname='1020'):
    """
    returns the dataframe with new columns of stamp, norm stamp
    """
    df["stamp"] = None
    df["norm_stamp"] = None
    
    fpath = (
        "/lustre/lrspec/ZTF_data/2022/1020/"
    )
    
    for filename, group in df.groupby('filename', sort=False):  # sort=False saves time
        path = fpath + filename + 'fits.fz'
        
        with fits.open(path, memmap=True) as hdul:  # Memory-map the FITS file
            indices = group.index.tolist()  # Batch access
            for idx in indices:
                df.at[idx, "stamp"] = get_stamp_from_hdul(hdul, idx, direcname, df, norm=False)
                df.at[idx, "norm_stamp"] = get_stamp_from_hdul(hdul, idx, direcname, df, norm=True)
    
    return df

def add_stamps3(df, direcname='1020'):
    """
    returns the dataframe with new columns of stamp, norm stamp
    """
    df["stamp"] = None
    df["norm_stamp"] = None
    
    fpath = (
        "/lustre/lrspec/ZTF_data/2022/1020/"
    )
    
    for filename, group in df.groupby('filename', sort=False):
        path = fpath + filename + 'fits.fz'
        
        with fits.open(path, memmap=True) as hdul:
            # Group by quad to minimize image normalizations
            for quad, quad_group in group.groupby('quad'):
                quad_data = hdul[int(quad)].data
                
                # Calculate normalization factors once per quad
                data_min = np.min(quad_data)
                data_max = np.max(quad_data)
                data_range = data_max - data_min
                
                for idx in quad_group.index:
                    row = int(df.at[idx, "row_from_nn"]) + 10
                    col = int(df.at[idx, "col_from_nn"]) + 10
                    
                    # Extract stamp
                    stamp = quad_data[row - 10 : row + 10, col - 10 : col + 10].copy()
                    df.at[idx, "stamp"] = stamp
                    
                    # Normalized stamp
                    if data_range > 0:
                        norm_stamp = (stamp - data_min) / data_range
                    else:
                        norm_stamp = stamp.copy()
                    df.at[idx, "norm_stamp"] = norm_stamp
                
                # Clear reference to help GC
                del quad_data
    
    return df

def add_normsobel_stuff(df, eff=False):
    df['sobel_norm_stdv_rat'] = np.zeros(len(df))
    df['sobel_norm_diffmax'] = np.zeros(len(df))
    
    

    for i, row in df.iterrows():
        if eff:
            normstamp = get_stamp_from_df_eff(i, df, norm=True)
        else:
            normstamp = sobels_to_array(row['norm_stamp'])
        
        xnorm, ynorm = justxy(normstamp)
        df['sobel_norm_stdv_rat'].loc[i] = np.std(xnorm)/np.std(ynorm)
        df['sobel_norm_diffmax'].loc[i] = np.max(xnorm)/np.max(ynorm)
        del normstamp, xnorm, ynorm
        gc.collect()
                                                     
          
    return df


def sobels_to_array(original):
    # array = np.zeros((20,20))
    cleaned = original[1:-1]
    
    # Split into rows
    rows = []
    current_row = []
    
    # Split by brackets to identify rows
    parts = cleaned.split('[')
    for part in parts:
        if not part.strip():
            continue
        
        # Remove closing bracket and split by whitespace
        part = part.replace(']', '')
        values = part.split()
        
        for val in values:
            val = val.strip()
            if val and val != ',':
                try:
                    current_row.append(float(val))
                except ValueError:
                    continue
        
        if len(current_row) == 20:
            rows.append(current_row)
            current_row = []
    
    return np.array(rows)
        
# row['sobels_x'].split()

def justxy(stamp):
    Lx = sobel(stamp, axis=1)  # column axis
    Ly = sobel(stamp, axis=0)  # rows axis
    # sobel_ratio = np.max(Lx) / np.max(Ly)  # we want this to be as large as possible
    return Lx, Ly



def inspect_stamp_and_map(df, index, eff=True, neweff="new_eff_data/", posneg=None, size=10, column=None, nn_or_imp="nn", thr=True, varthresh=.9999,star=False, direcname='1020', pdirec="predsarrays"):
    '''
    for efficiency data
    '''
    wid = 10
    if column!=None:
        index = df[df[column]==index].index[0]
    if eff==True:
        if posneg==None:
            posneg = df.loc[index, "type"]
            print("Need posneg if you want to see the correct pimg...")
        if posneg=="pos":
            extension = "_1_implanted_pos.npy"
            if neweff=="new_eff_data/" or neweff=="nosat_eff_data/":
                extension = "_"+str(df.loc[index, "quad"])+".npy"
            
            fpath = '/lustre/lrspec/ZTF_data/efficiency_testing/'+neweff+'full_image_implantations_pos/images/'
            ppath = "/lustre/lrspec/ZTF_data/efficiency_testing/"+neweff+"full_image_implantations_pos/"+pdirec+"/"

        elif posneg=="neg":
            extension = "_1_implanted_neg.npy"
            if neweff=="new_eff_data/" or neweff=="nosat_eff_data/":
                extension = "_"+str(df.loc[index, "quad"])+".npy"
            
            fpath = '/lustre/lrspec/ZTF_data/efficiency_testing/'+neweff+'full_image_implantations_neg/images/'
            ppath = "/lustre/lrspec/ZTF_data/efficiency_testing/"+neweff+"full_image_implantations_neg/"+pdirec+"/"
        
        else:
            print("POSNEG ISSUE")
        fname = df.loc[index]["filename"] + extension
    else:
        posneg = df.loc[index, 'type']

    if nn_or_imp=="nn":
        fig, ax = plt.subplots(1, 2, figsize=(7,2))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(10,2))
    
    # print(df.iloc[index]["filename"])
    if nn_or_imp == "nn":
        row = int(df.loc[index]["row_from_nn"])+wid
        col = int(df.loc[index]["col_from_nn"])+wid
    elif nn_or_imp == "imp":
        row = int(df.loc[index]["col_from_implant"])
        col = int(df.loc[index]["row_from_implant"])
        
    if eff:
        path = fpath + fname 
        array = np.load(path)
        imj = ax[0].imshow(
                    
                        array[
                            row - size : row + size ,
                            col  - size : col  + size,
                        ]
                    
                    )
    else:
        imj = ax[0].imshow(sobels_to_array(df.loc[index, 'stamp']))
                    
    nrow, ncol = 3080, 3072 
    if eff==True:
        if pdirec=="predsarrays":
            if posneg =="pos":
                pfile = df.loc[index]["filename"]+"_extraction_predsarray_pos_predsarray.npy"
                predsarray = np.load(ppath+pfile)
                if df.loc[index, "type"]=="pos":
                    pimg = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
                else:
                    pimg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
            elif posneg=="neg":
                pfile = df.loc[index]["filename"]+"_extraction_predsarray_neg_predsarray.npy"
                predsarray = np.load(ppath+pfile)
                if df.loc[index, "type"]=="pos":
                    pimg = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
                else:
                    pimg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
        else:
            pfile = df.loc[index]["filename"]+"_extraction_predsarray_1_predsarray.npy"
            if neweff=="new_eff_data/" or neweff=="nosat_eff_data/":
                pfile = df.loc[index]["filename"]+"_" + str(df.loc[index]["quad"])+"_predsarray.npy"
            predsarray = np.load(ppath+pfile)
            if df.loc[index, "type"]=="pos":
                pimg = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
            else:
                pimg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
    else:      
        path = '/lustre/lrspec/ZTF_data/science_extractions/'+direcname+'/'
        pred = df.loc[index]["filename"]
        extra = '_extraction_predsarray_'+str(df.loc[index,'quad'])+'_predsarray.npy'
        print(path+pred+extra)
        predsarray = np.load(path+pred+extra)
        if df.loc[index, "type"]=="pos":
            pimg = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
        else:
            pimg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)





    if thr:
        ax[1].imshow(
        
                    pimg[
                        col - size  - wid : col + size - wid,
                        row  - size - wid: row  + size - wid,
                    ].T>varthresh
                )
    else:
        ax[1].imshow(
        
                    pimg[
                        col - size  - wid: col + size - wid,
                        row  - size - wid: row  + size - wid,
                    ].T
                )
        ax[2].imshow(
        
                    pimg[
                        col - size  - wid: col + size - wid,
                        row  - size - wid : row  + size - wid,
                    ].T>varthresh
                )
        ax[2].set_title("CNN Thresholded")
    if star==True:
                    starpos = df.loc[index]['closest_x']
                    ax[0].scatter(starpos- (col - size  + wid), 10)
#     pfile_pos = df.iloc[index]["filename"]+"_extraction_predsarray_pos_predsarray.npy"
#     predsarray_pos = np.load(ppath+pfile_pos)
#     pimg_pos = predsarray_pos[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
#     pfile_neg = df.iloc[index]["filename"]+"_extraction_predsarray_neg_predsarray.npy"
#     predsarray_neg = np.load(ppath+pfile_neg)
#     pimg_neg = predsarray_neg[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
#     ax[1].imshow(
        
#                     pimg_neg[
#                         col - size : col + size,
#                         row  - size : row  + size,
#                     ].T>.9999
#                 )
#     ax[2].imshow(pimg_pos[
#                         col - size : col + size,
#                         row  - size : row  + size,
#                     ].T>.9999
#                 ) 
    if eff==True:
        ax[0].annotate(
        "Injection",
        xy=(  df["row_from_implant"].loc[index]-col+size,df["col_from_implant"].loc[index]-row+size),
        xytext=(size+3*size/4, size/2),  
        arrowprops=dict(
            facecolor="red",
            edgecolor="red",
            # shrink=1.0,
            # headlength=5,
            # headwidth=3,
        ),
        color="white",
        fontsize=10,
        ha="center",
        # va="top",
        rotation=0,
        zorder=10,
    )
        if np.abs(df['dist_to_implant'].loc[index])<30:
            ax[0].set_title("Injected image, Index "+str(index)+ ", A = "+str(df["A_from_implant"].loc[index])+", sig = "+str(df["sig_from_implant"].loc[index]))
        else:
            ax[0].set_title("No nearby injections")
    else:
        ax[0].set_title("Injected image, Index "+str(index))
        
    ax[1].set_title("CNN "+str(df.loc[index, "type"])+" prediction")
    # ax[2].set_title("CNN pos prediction")
    # print(df["row_from_implant"].iloc[index], col, df["col_from_implant"].iloc[index], row)
    fig.colorbar(imj, orientation='horizontal')
    plt.show()
    
    
def normarray(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))
    
def inspect_df_w_stamp_color(df, imgsize=10, star=False, clim=None, plot="stamp", norm=False, titlecol=[], twodec=False, plot_centers=False):
    """
    plots the stamps corresponding to detections in a dataframe
    no limit, so put in df[:100] or something probably to save time
    """
    # Define a color palette for different elements
    colors = ['red', 'blue', 'green', 'darkorange', 'purple', 'brown', 'deeppink', 'gray', 'olive', 'cyan']
    
    wid = 10
    figcols = 5
    figrows = 1 + len(df) // figcols
    fig, ax = plt.subplots(figrows, figcols, figsize=(figcols * 1.8, figrows * 1.8))
    for i in range(figrows):
        for j in range(figcols):
            index = figcols * i + j
            if index < len(df):
                fname = df.iloc[index]["filename"] + "fits.fz"
                row = int(df.iloc[index]["row_from_nn"])
                col = int(df.iloc[index]["col_from_nn"])
                quad = int(df.iloc[index]["quadrant"])
                # stamp = df.iloc[index][plot]
                stamp, bg = get_stamp_from_df(index, df, direcname="2022/1020/", norm=True, quad=quad, addrow=0, addcol=0, iloc=True, imgsize=imgsize)
                
                if isinstance(stamp, str):
                    stamp = sobels_to_array(stamp)
                
                if norm:
                    stamp = normarray(stamp)
                
                ax[i, j].imshow(stamp, vmax=clim)
                
                if star==True:
                    starpos = df.iloc[index]['closest_x']
                    ax[i, j].scatter(starpos - (col - imgsize + wid), 10)
                if plot_centers:
                    center = df.iloc[index]['row_from_nn'] + 10
                    ax[i, j].scatter(10, center - (row - imgsize + wid))
                    center2 = df.iloc[index]['central_point_brightening']
                    ax[i, j].scatter(10, center2 - (row - imgsize + wid))
                
                ax[i, j].set_axis_off()
                
                if titlecol==None or titlecol==[]:
                    ax[i, j].set_title(str(index))
                else:
                    # Build title with proper spacing between colored values
                    num_values = len(titlecol)
                    
                    # Calculate spacing - distribute evenly across the width
                    if num_values == 1:
                        positions = [0.5]
                    else:
                        # Space them out more for better separation
                        spacing = 0.8 / (num_values - 1)  # Use 80% of width
                        positions = [0.1 + i * spacing for i in range(num_values)]
                    
                    for idx, title in enumerate(titlecol):
                        if twodec==False:
                            if type(df.iloc[index][title])!=str:
                                value = round(df.iloc[index][title])
                            else:
                                value = df.iloc[index][title]
                        
                        else:
                            if type(df.iloc[index][title])!=str:
                                value = round(df.iloc[index][title]*100)/100
                            else:
                                value = df.iloc[index][title]
                        
                        color = colors[idx % len(colors)]
                        
                        # Place colored text at calculated position
                        ax[i, j].text(positions[idx], 1.02, str(value),
                                     transform=ax[i, j].transAxes,
                                     ha='center', va='bottom',
                                     fontsize=10, color=color, weight='bold')
    
    # Create color-coded suptitle legend
    if titlecol!=None and titlecol!=[]:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[idx % len(colors)], 
                                label=title) 
                          for idx, title in enumerate(titlecol)]
        fig.legend(handles=legend_elements, loc='upper center', 
                  ncol=len(titlecol), frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    
    
    

# def add_satellite_flag(df, new=True):
#     df['satellite_flag_2_2_new'] = np.zeros(len(df))
#     df['satellite_flag_1_7_new'] = np.zeros(len(df))
#     df['satellite_flag'] = np.zeros(len(df))
#     if new==True:
#         for i, row in df.iterrows():            
#             if row['max_num_consecutive']>2 and row['snr_global_new']>2:
#                 df.at[i, 'satellite_flag_2_2_new']=1
#             if row['max_num_consecutive']>1 and row['snr_global_new']>7:
#                 df.at[i, 'satellite_flag_1_7_new']=1
#             if df.at[i, 'satellite_flag_2_2_new']==1 and df.at[i, 'satellite_flag_1_7_new']==1:
#                 df.at[i, 'satellite_flag']=1
#     else:
#         for i, row in df.iterrows():            
#             if row['max_num_consecutive']>2 and row['snr_global']>2:
#                 df.at[i, 'satellite_flag_2_2_new']=1
#             if row['max_num_consecutive']>1 and row['snr_global']>7:
#                 df.at[i, 'satellite_flag_1_7_new']=1
#             if df.at[i, 'satellite_flag_2_2_new']==1 and df.at[i, 'satellite_flag_1_7_new']==1:
#                 df.at[i, 'satellite_flag']=1
#     return df


#looking for multiple transients that progress
#something like - num-consecutive

def add_num_consecutive(df, df_larger=None):
    '''
    use df_larger if you want to run this on a subset (df)
    '''
    df = df.copy() 
    if type(df_larger)!=pd.core.frame.DataFrame:
        print("no df_larger")
        df_larger=df
    df["num_in_file_new"] = np.zeros(len(df))
    df["num_consecutive_up_new"] = np.zeros(len(df))
    df["num_consecutive_down_new"] = np.zeros(len(df))
    df["possible_up_indices_new"] = np.zeros(len(df), dtype=object)
    df["possible_down_indices_new"] = np.zeros(len(df), dtype=object)
    df["max_num_consecutive_new"] = np.zeros(len(df))
    for fname in np.unique(df["filename"]):
        for quad in np.unique(df[df["filename"]==fname]["quadrant"]):
            # for polarity in np.unique(df[(df["filename"]==fname) * (df["quadrant"]==quad)]['type']):
            # for i in [1]: #listen i just dont want to get rid of the tab
            subsection = df[(df["filename"]==fname) * (df["quadrant"]==quad)]#* (df["type"]=='pos')]
            # subsection = subsection[subsection['detection_size']>10]
            subsection_of_larger = df_larger[(df_larger["filename"]==fname) * (df_larger["quadrant"]==quad)* (df_larger["sobel_norm_diffmax"]<10)]
            num_in_file_here = len(df_larger[(df_larger["filename"]==fname) * (df_larger["quadrant"]==quad)]) 
            # subsection_of_larger = subsection_of_larger[subsection_of_larger['detection_size']>10]
            
            if len(subsection_of_larger)<2:
                    continue #leaves num_consecutive at zero
            if len(subsection)<1: #this is for if we use a subsection)
                continue
            # print(len(subsection))
            # print("HEY!")
                                             
            for index in subsection.index:
                df.at[index, "num_in_file_new"]=num_in_file_here
                # print(index)
                # print(num_in_file_here)
                # print(df.loc[index]["num_in_file"])
                
                original_x = df.loc[index]["closest_x"] #x
                original_row = df.loc[index]["row_from_nn"] #y
                num_going_up = 0
                num_going_down = 0
                possible_up_indices = []
                possible_down_indices = []
                for i, rrow in subsection_of_larger.iterrows():
                    # print(rrow["col_from_nn"])
                    # print(original_col)
                    # if (rrow["col_from_nn"] > original_col) * (rrow["row_from_nn"] > original_row): #if both true
                    #     num_going_up+=1
                    #     possible_up_indices.append(i)
                    # elif (rrow["col_from_nn"] < original_col) * (rrow["row_from_nn"] < original_row):
                    #     num_going_up+=1
                    #     possible_up_indices.append(i)
                    # elif (rrow["col_from_nn"] < original_col) * (rrow["row_from_nn"] > original_row):
                    #     num_going_down+=1
                    #     possible_down_indices.append(i)
                    # elif (rrow["col_from_nn"] > original_col) * (rrow["row_from_nn"] < original_row):
                    #     num_going_down+=1
                    #     possible_down_indices.append(i)
                    if (rrow["closest_x"] > original_x) * (rrow["row_from_nn"] > original_row): #if both true
                        num_going_up+=1
                        possible_up_indices.append(i)
                    elif (rrow["closest_x"] < original_x) * (rrow["row_from_nn"] < original_row):
                        num_going_up+=1
                        possible_up_indices.append(i)
                    elif (rrow["closest_x"] < original_x) * (rrow["row_from_nn"] > original_row):
                        num_going_down+=1
                        possible_down_indices.append(i)
                    elif (rrow["closest_x"] > original_x) * (rrow["row_from_nn"] < original_row):
                        num_going_down+=1
                        possible_down_indices.append(i)
                
                #okay so first i want to drop ones where they're in the same star
                df_going_up = subsection_of_larger.loc[possible_up_indices].drop_duplicates(subset="closest_x")
                df_going_down = subsection_of_larger.loc[possible_down_indices].drop_duplicates(subset="closest_x")
                df.at[index, "max_num_consecutive_new"] = np.max([len(df_going_up), len(df_going_down)])
                del df_going_up
                del df_going_down
                gc.collect()
                # print(possible_down_indices)
                df.at[index, "num_consecutive_up_new"] = num_going_up
                df.at[index, "num_consecutive_down_new"] = num_going_down
                df.at[index, "possible_down_indices_new"] = np.array(possible_down_indices)
                df.at[index, "possible_up_indices_new"] = np.array(possible_up_indices)
    return df
                
                
        
        
def find_dups(df):
    '''
    find duplicate transient candidates
    '''
    df["dup_indices"] = np.zeros(len(df), dtype=object)
    df['is_smaller_dup'] = np.zeros(len(df)) 
    for fname in np.unique(df["filename"]):
        for quad in np.unique(df[df["filename"]==fname]["quadrant"]):
            subsection = df[(df["filename"]==fname) * (df["quadrant"]==quad)]
            for polarity in ['pos','neg']:
                for index in subsection[subsection['type']==polarity].index:
                    width = 20 #+ np.round(np.sqrt(subsection.loc[index]['detection_size'])) #removed for better efficiency
                    original_col = subsection.loc[index]["col_from_nn"] #x
                    original_row = subsection.loc[index]["row_from_nn"] #y
                    original_size = subsection.loc[index]["detection_size"]
                    dup_indices = []
                    for i, row in subsection[subsection['type']==polarity].iterrows():
                        if np.abs(original_col-row["col_from_nn"])<=width and np.abs(original_row-row["row_from_nn"])<=width:
                            if i!=index:
                                dup_indices.append(i)
                            if row["detection_size"]<original_size:
                                df.at[i, 'is_smaller_dup']=1
                            elif row["detection_size"]==original_size:
                                if np.abs(row['Diff'])>np.abs(subsection.loc[index]["Diff"]):
                                    df.at[i, 'is_smaller_dup']=1
                                elif np.abs(row['Diff'])==np.abs(subsection.loc[index]["Diff"]):
                                    if i>index: #this one is just random
                                        df.at[i, 'is_smaller_dup']=1
                    df.at[index, "dup_indices"] = np.array(dup_indices)
    return df


def add_num_in_file_in_star(df):
    df['num_in_file_in_star']=np.zeros(len(df))
    for filename, group in df.groupby('filename'):     
        for quadnum, quadgroup in group.groupby('quadrant'):
            for starx, stargroup in quadgroup.groupby('closest_x'):
                num_in_file_in_star = len(stargroup)
                for index in stargroup.index:
                    df.at[index, 'num_in_file_in_star']=num_in_file_in_star
    return df


def add_sat_and_proximity(df, eff=False):
    df = df.copy()
    df['sat_flag'] = np.zeros(len(df))
    df['sat_neighbor'] = np.zeros(len(df))
    for i, row in df.iterrows():
        if eff==True:
            if ((row['sobel_norm_diffmax']<5)*(row['max_num_consecutive_new']>1)*(row['snr_local']>=2)):
                df.loc[i,'sat_flag']=1
        else:
            if ((row['sobel_norm_diffmax']<5)*(row['max_num_consecutive_new']>1)*(row['snr_local_new']>=2)):
                df.loc[i,'sat_flag']=1
    for i, satrow in df[df['sat_flag']==1].iterrows():
        fname = satrow['filename']
        quad = satrow['quadrant']
        subsection = df[(df["filename"]==fname) * (df["quadrant"]==quad)]
        original_col = satrow["col_from_nn"]
        original_row = satrow["row_from_nn"]
        for i, neighborrow in subsection.iterrows():
            if (np.abs(neighborrow["col_from_nn"] - original_col)<40) * (np.abs(neighborrow["row_from_nn"] - original_row)<40):
                df.loc[i,'sat_neighbor']=1
    return df

def match_stars_eff(df):
    closest_xs = np.zeros(len(df))
    closest_ys= np.zeros(len(df))
    Diffs = np.zeros(len(df))
    brightness_ranking = np.zeros(len(df))
    xy_path = "/lustre/lrspec/ZTF_data/corrected_coordinates/xy_arrays/"
    quad = str(1)
    i=0
    for index, rrow in df.iterrows():

        xy_file = xy_path+df.loc[index]['filename'][:35]+quad+".fits"
        with fits.open(xy_file) as hdul_xy:
            col = rrow["col_from_nn"]
            target_x = col+10
            # print(xy_file)
            data = hdul_xy[1].data  
            stored_x = data["X"][:25]  # Column "X"
            stored_y = data["Y"][:25]
            closest_idx = np.argmin(np.abs(stored_x - target_x))
            
            closest_x, closest_y = stored_x[closest_idx], stored_y[closest_idx]
            closest_xs[i]=closest_x
            closest_ys[i]=closest_y
            Diffs[i]=stored_x[closest_idx] - target_x
            brightness_ranking[i]= closest_idx
            # print(closest_idx)
            # print(df.loc[index]['brightness_ranking'])
            i+=1 #yes there is a better way to do this
    df['closest_x'] = closest_xs
    df['closest_y']= closest_ys
    df['Diff'] = Diffs
    df['brightness_ranking'] = brightness_ranking
    return df


def make_obj_list(predsarray, abst_thr, fname, quad):
    '''
    get dataframe from cnn predictiions
    '''
    nrow, ncol = 3080, 3072
    wid=10
    pimg_pos = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
    # pimg_bg = predsarray[:, 0].reshape(ncol - 2 * wid, nrow - 2 * wid)
    # pimg_streak = predsarray[:, 4].reshape(ncol - 2 * wid, nrow - 2 * wid)
    pimg_neg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
    pimg_cr = predsarray[:, 3].reshape(ncol - 2 * wid, nrow - 2 * wid)
    pimg_sat = predsarray[:, 5].reshape(ncol - 2 * wid, nrow - 2 * wid)

    # pimg_both = pimg_pos + pimg_neg

    structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]  # for what counts as one object

    # FOR POS
    thrdet = pimg_pos > abst_thr
    threshed = thrdet.T  # it takes the transposed pimg

    labs, nlabs = snm.label(threshed, structure)
    loc_list_pos = snm.center_of_mass(threshed, labs, range(1, nlabs + 1))
    loc_list_pos = np.array(loc_list_pos)
    objsizes_pos = pr.get_objsizes(labs, nlabs)

    fnames = np.full(len(loc_list_pos), fname)
    poses = np.full(len(loc_list_pos), "pos")
    if len(loc_list_pos) > 0:
        poslist = (
            np.vstack(
                (fnames, poses, loc_list_pos[:, 0], loc_list_pos[:, 1], objsizes_pos)
            )
        ).T
    else:
        poslist = 0

    # FOR NEG
    thrdet = pimg_neg > abst_thr
    threshed = thrdet.T  # it takes the transposed pimg

    labs, nlabs = snm.label(threshed, structure)
    loc_list_neg = snm.center_of_mass(threshed, labs, range(1, nlabs + 1))
    loc_list_neg = np.array(loc_list_neg)

    objsizes_neg = pr.get_objsizes(labs, nlabs)

    fnames = np.full(len(loc_list_neg), fname)
    negs = np.full(len(loc_list_neg), "neg")

    if len(loc_list_neg) > 0:
        neglist = (
            np.vstack(
                (fnames, negs, loc_list_neg[:, 0], loc_list_neg[:, 1], objsizes_neg)
            )
        ).T
    else:
        neglist = 0
    # FOR CRS
    thrdet = pimg_cr > abst_thr
    threshed = thrdet.T  # it takes the transposed pimg

    labs, nlabs = snm.label(threshed, structure)
    loc_list_cr = snm.center_of_mass(threshed, labs, range(1, nlabs + 1))
    loc_list_cr = np.array(loc_list_cr)

    objsizes_cr = pr.get_objsizes(labs, nlabs)

    fnames = np.full(len(loc_list_cr), fname)
    if len(loc_list_cr) > 0:
        crs = np.full(len(loc_list_cr), "cr")
        # print(np.shape(fnames))
        # print(np.shape(crs))
        # print(np.shape(loc_list_cr[:,0]))
        # print(np.shape(loc_list_cr[:,1]))
        # print(np.shape(objsizes_cr))
        crlist = (
            np.vstack((fnames, crs, loc_list_cr[:, 0], loc_list_cr[:, 1], objsizes_cr))
        ).T
    else:
        crlist = 0
#     # FOR BOTH
#     thrdet = pimg_both > abst_thr
#     threshed = thrdet.T  # it takes the transposed pimg

#     labs, nlabs = snm.label(threshed, structure)
#     loc_list_both = snm.center_of_mass(threshed, labs, range(1, nlabs + 1))
#     loc_list_both = np.array(loc_list_both)

#     objsizes_both = pr.get_objsizes(labs, nlabs)

#     fnames = np.full(len(loc_list_both), fname)
#     boths = np.full(len(loc_list_both), "both")
#     if len(loc_list_both) > 0:
#         bothlist = (
#             np.vstack(
#                 (fnames, boths, loc_list_both[:, 0], loc_list_both[:, 1], objsizes_both)
#             )
#         ).T
    thrdet = pimg_sat > abst_thr
    threshed = thrdet.T  # it takes the transposed pimg

    labs, nlabs = snm.label(threshed, structure)
    loc_list_sat = snm.center_of_mass(threshed, labs, range(1, nlabs + 1))
    loc_list_sat = np.array(loc_list_sat)

    objsizes_sat = pr.get_objsizes(labs, nlabs)

    fnames = np.full(len(loc_list_sat), fname)
    if len(loc_list_sat) > 0:
        sats = np.full(len(loc_list_sats), "sat")
        # print(np.shape(fnames))
        # print(np.shape(crs))
        # print(np.shape(loc_list_cr[:,0]))
        # print(np.shape(loc_list_cr[:,1]))
        # print(np.shape(objsizes_cr))
        satlist = (
            np.vstack((fnames, sats, loc_list_sat[:, 0], loc_list_sat[:, 1], objsizes_sat))
        ).T
    else:
        satlist = 0

    # del pimg_cr
    # del pimg_pos
    # del pimg_neg
    # # del pimg_bg
    # del pimg_sat
    # del thrdet
    # del loc_list_pos
    # del loc_list_neg
    # del objsizes_pos
    # del objsizes_neg
    # del labs
    # del threshed
    gc.collect()

    # np.save(path, n_loc_size_ray_b)
    if isinstance(poslist, np.ndarray):
        listlist = poslist
        for listype in [neglist, crlist, satlist]:
            if isinstance(listype, np.ndarray):
                listlist = np.vstack((listlist, listype))
        alllist = pd.DataFrame(listlist)
    elif isinstance(neglist, np.ndarray):
        listlist = neglist
        for listype in [poslist, crlist, satlist]:
            if isinstance(listype, np.ndarray):
                listlist = np.vstack((listlist, listype))
        alllist = pd.DataFrame(listlist)
    else:
        print("NO DETECTIONS IN THIS FILE")
        alllist = None
        return alllist
    
    

    alllist.rename(
        columns={
            0: "filename",
            1: "type",
            2: "row_from_nn",
            3: "col_from_nn",
            4: "detection_size",
        },
        inplace=True,
    )
    alllist['quad'] = np.full(len(alllist), quad)

    alllist[["row_from_nn", "col_from_nn", "detection_size"]] = alllist[
        ["row_from_nn", "col_from_nn", "detection_size"]
    ].apply(pd.to_numeric)
    # np.save(path, n_loc_size_ray_b)
    return alllist


def save_predsarrays(image, model, wid, path):
    """
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
    so a list of tuples, then an int, then the probability that its column is a streak
    to flag as a probable cosmic ray
    """

    nrow, ncol = 3080, 3072
    print("getting postage stamps")
    stamps = gpsf.get_postage_stamps_full(image, 10)
    stampbatches = np.array_split(
        stamps, stamps.shape[0] // 20000
    )  # (image.shape[0] - 2 * wid) )
    # I WANNA CHANGE THIS TO SPEED HIM UP
    # so to be faster there should be less batches
    # so i want the second argument to be smaller
    # so i want the denomitor of the second argument to be larger
    preds = []
    print("getting predictions:")
    for stamps in stampbatches:#tqdm(           # getting all the predictions. does not renormalize
        pred_l = model.predict_on_batch(stamps)
        preds.append(pred_l)
    predsarray = np.concatenate(preds)
    # i would like to save the predsarrays i think. for reproducibility
    np.save(path + "_predsarray", predsarray)  
    del stamps
    del stampbatches
    del preds
    del predsarray
    gc.collect()
    return  # predsarray

def match_implantations_eff(df_original, posneg, neweff=""):
    df = df_original.copy()
    if posneg=="neg":
        extension="_neg"
    elif posneg=="pos":
        extension="_pos"
    row_from_implant = {}
    col_from_implant = {}
    A_from_implant = {}
    sig_from_implant = {}
    dist_to_implant = {}
    # i=0
    for (filename, quad), group in df.groupby(['filename', 'quad']):   
        # for quad in np.unique(group['quad']):
            # infopath = (
            #     "/lustre/lrspec/ZTF_data/efficiency_testing/"+neweff+"full_image_implantations_"+posneg+"/info/"
            #     + str(filename)
            #     + "_"
            #     + str(1)
            #     + "_implanted_info"+extension+".npy"
            # )
        infopath = (
            "/lustre/lrspec/ZTF_data/efficiency_testing/"+neweff+"full_image_implantations_"+posneg+"/info/"
            + str(filename)
            + "_"
            + str(quad)+"_info.npy"
        )
        info = np.load(infopath)
        for idx in group.index:
            distance_y = np.abs(df.loc[idx,"row_from_nn"] + 10 - info[:,1])
            distance_x = np.abs(df.loc[idx,"col_from_nn"] + 10 - info[:,0])
            total = np.sqrt(distance_y**2 + distance_x**2)
            closest_idx_info = np.argmin(total)
            dist_to_implant[idx] = total[closest_idx_info]
            row_from_implant[idx] = info[closest_idx_info][1]
            col_from_implant[idx] = info[closest_idx_info][0]
            A_from_implant[idx] = info[closest_idx_info][2]
            sig_from_implant[idx] = info[closest_idx_info][3]
            
    df["dist_to_implant"] = df.index.map(dist_to_implant)
    df["row_from_implant"] = df.index.map(row_from_implant)
    df["col_from_implant"] = df.index.map(col_from_implant)
    df["A_from_implant"] = df.index.map(A_from_implant)
    df["sig_from_implant"] = df.index.map(sig_from_implant)
    return df


def inspect_stamp_and_all_maps(df, index, eff=True, posneg=None, size=10, column=None, nn_or_imp="nn", thr=True, star=False, override=False):
    '''
    for efficiency data
    '''
    pdirecs = ["predsarrays/",
                "presdarrays_flipped/",
                "presdarrays_both/",
                "predsarrays_dup/"]
    names = ['Original', 'Flipped', 'Both', 'Duplicate']
    ppaths = []
    wid = 10
    if column!=None:
        index = df[df[column]==index].index[0]
    if eff==True:
        
        if posneg==None:
            posneg = df.loc[index, "type"]
            print("Need posneg if you want to see the correct pimg...")
        if posneg=="pos":
            extension = "_1_implanted_pos.npy"
            fpath = '/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_pos/images/'
            for pdirec in pdirecs:
                ppaths.append("/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_pos/"+pdirec)

        elif posneg=="neg":
            extension = "_1_implanted_neg.npy"
            fpath = '/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_neg/images/'
            for pdirec in pdirecs:
                ppaths.append("/lustre/lrspec/ZTF_data/efficiency_testing/full_image_implantations_neg/"+pdirec)
        
        else:
            print("POSNEG ISSUE")
        fname = df.loc[index]["filename"] + extension
    else:
        posneg = df.loc[index, 'type']


    fig, ax = plt.subplots(1, 5, figsize=(15,2))
    
    # print(df.iloc[index]["filename"])
    if nn_or_imp == "nn":
        row = int(df.loc[index]["row_from_nn"])+wid
        col = int(df.loc[index]["col_from_nn"])+wid
    elif nn_or_imp == "imp":
        row = int(df.loc[index]["col_from_implant"])
        col = int(df.loc[index]["row_from_implant"])
        
    if eff:
        path = fpath + fname 
        array = np.load(path)
        imj = ax[0].imshow(
                    
                        array[
                            row - size : row + size ,
                            col  - size : col  + size,
                        ]
                    
                    )
    else:
        imj = ax[0].imshow(sobels_to_array(df.loc[index, 'stamp']))
                    
    nrow, ncol = 3080, 3072 
    pimgs = []
    for i, pdirec in enumerate(pdirecs):
        if pdirec=="predsarrays/":
            if posneg =="pos":
                pfile = df.loc[index]["filename"]+"_extraction_predsarray_pos_predsarray.npy"
                predsarray = np.load(ppaths[i]+pfile)
                if override==False:
                    if df.loc[index, "type"]=="pos":
                        pimg = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
                    else:
                        pimg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
                elif override=="pos":
                    pimg = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
                elif override=="neg":
                    pimg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
            elif posneg=="neg":
                pfile = df.loc[index]["filename"]+"_extraction_predsarray_neg_predsarray.npy"
                predsarray = np.load(ppaths[i]+pfile)
                if override==False:
                    if df.loc[index, "type"]=="pos":
                        pimg = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
                    else:
                        pimg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
                elif override=="pos":
                    pimg = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
                elif override=="neg":
                    pimg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
                    
        else:
            pfile = df.loc[index]["filename"]+"_extraction_predsarray_1_predsarray.npy"
            predsarray = np.load(ppaths[i]+pfile)
            if override==False:
                if df.loc[index, "type"]=="pos":
                    pimg = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
                else:
                    pimg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
            elif override=="pos":
                pimg = predsarray[:, 1].reshape(ncol - 2 * wid, nrow - 2 * wid)
            elif override=="neg":
                pimg = predsarray[:, 2].reshape(ncol - 2 * wid, nrow - 2 * wid)
        pimgs.append(pimg)

    for num, pimg in enumerate(pimgs):
        if thr:
            ax[num+1].imshow(
            
                        pimg[
                            col - size  - wid : col + size - wid,
                            row  - size - wid: row  + size - wid,
                        ].T>.9999, cmap='viridis'
                    )
        else:
            ax[num+1].imshow(
            
                        pimg[
                            col - size  - wid: col + size - wid,
                            row  - size - wid: row  + size - wid,
                        ].T, cmap='viridis'
                    )
        if override==False:
            ax[num+1].set_title(names[num]+" CNN "+str(df.loc[index, "type"])+" prediction")
        else:
            ax[num+1].set_title(names[num]+" CNN "+override+" prediction")
    if star==True:
                    starpos = df.loc[index]['closest_x']
                    ax[0].scatter(starpos- (col - size  + wid), 10)

#                 ) 
    if eff==True:
        ax[0].annotate(
        "Injection",
        xy=(  df["row_from_implant"].loc[index]-col+size,df["col_from_implant"].loc[index]-row+size),
        xytext=(size+3*size/4, size/2),  
        arrowprops=dict(
            facecolor="red",
            edgecolor="red",
            # shrink=1.0,
            # headlength=5,
            # headwidth=3,
        ),
        color="white",
        fontsize=10,
        ha="center",
        # va="top",
        rotation=0,
        zorder=10,
    )
        if df.loc[index, 'dist_to_implant']<40:
            ax[0].set_title("Injected image, Index "+str(index)+ "\nA = "+str(df["A_from_implant"].loc[index])+", sig = "+str(df["sig_from_implant"].loc[index]))
        else:
            ax[0].set_title("Injected image, Index "+str(index)+"\nno nearby implantation")
    else:
        ax[0].set_title("Injected image, Index "+str(index))
        
    
    # ax[2].set_title("CNN pos prediction")
    # print(df["row_from_implant"].iloc[index], col, df["col_from_implant"].iloc[index], row)
    fig.colorbar(imj, orientation='horizontal')
    plt.show()
    
def show_confusion_matrix(save_name, sats=False, label_pred_te=None, targ_te=None, folds=False, title=None, absolute=False, mask=None):
    foldses = ["0","1","2","3","4"]
    if absolute==False:
        tonorm="true"
    elif absolute==True:
        tonorm=None
    # feat_te = np.load("/home/3155/saved_loss/targ_te_"+save_name+".npy")
    if sats==True:
        display_labels=["Background", "  Brightening", "Dimming", "Cosmic Ray", "Streak", "Satellite"]
    else:
        display_labels=["Background", "  Brightening", "Dimming", "Cosmic Ray", "Streak"]
    if type(targ_te)!=np.ndarray:
        targ_te = np.load("/home/3155/saved_loss/targ_te_"+save_name+".npy")
    if 1==1: #too lazy to de-indent
        print("loading data")
        if folds!=True:
            if type(label_pred_te)!=np.ndarray:
                if type(folds)==int:
                    label_pred_te = np.load("/home/3155/saved_loss/label_pred_te_"+save_name+"_fold_"+str(folds)+".npy")
                elif folds=='average':
                    label_pred_tes = []
                    for fold in foldses:
                        label_pred_tes.append(np.load("/home/3155/saved_loss/label_pred_te_"+save_name+"_fold_"+str(fold)+".npy"))
                    label_pred_te = np.average(np.array(label_pred_tes),axis=0)
                                              
                else:
                    label_pred_te = np.load("/home/3155/saved_loss/label_pred_te_"+save_name+".npy")
            if 1==1:
                    if type(mask)==np.ndarray:
                        label_pred_te = label_pred_te[mask]
                        targ_te = targ_te[mask]
                    disp = ConfusionMatrixDisplay.from_predictions(
                        targ_te,
                        np.argmax(label_pred_te, axis=1),
                        display_labels=display_labels,
                        cmap="bone",
                        normalize=tonorm,
                    )
                        # Adjust tick label size
                    disp.ax_.tick_params(axis="y", labelsize=17)
                    disp.ax_.tick_params(axis="x", labelsize=17, rotation=45)
                    # x.set_xticks([1, 2, 3], labels, rotation=45, ha='right')
                
                    # Adjust font size of numbers
                    for text in disp.ax_.texts:  # Access text annotations
                        text.set_fontsize(16)  # Set desired font size
                    disp.ax_.set_xlabel("Predicted Label", fontsize=18)  # Adjust X-axis label
                    disp.ax_.set_ylabel("True Label", fontsize=18)  # Adjust Y-axis label
                    if title==None:
                        title=save_name
                    disp.ax_.set_title(title, fontsize=20)
                
                    cbar = disp.im_.colorbar  # Access the colorbar
                    cbar.ax.tick_params(labelsize=16)  # Adjust tick label size
                    plt.show()
        elif folds==True:
            for fold in foldses:
                label_pred_te = np.load("/home/3155/saved_loss/label_pred_te_"+save_name+"_fold_"+fold+".npy")
                if type(mask)==np.ndarray:
                        label_pred_te = label_pred_te[mask]
                        targ_te = targ_te[mask]
                if 1==1:
                    disp = ConfusionMatrixDisplay.from_predictions(
                        targ_te,
                        np.argmax(label_pred_te, axis=1),
                        display_labels=display_labels,
                        cmap="bone",
                        normalize=tonorm,
                    )
                        # Adjust tick label size
                    disp.ax_.tick_params(axis="y", labelsize=17)
                    disp.ax_.tick_params(axis="x", labelsize=13)
                
                    # Adjust font size of numbers
                    for text in disp.ax_.texts:  # Access text annotations
                        text.set_fontsize(16)  # Set desired font size
                    disp.ax_.set_xlabel("Predicted Label", fontsize=14)  # Adjust X-axis label
                    disp.ax_.set_ylabel("True Label", fontsize=18)  # Adjust Y-axis label
                    disp.ax_.set_title(save_name)
                
                    cbar = disp.im_.colorbar  # Access the colorbar
                    cbar.ax.tick_params(labelsize=16)  # Adjust tick label size
                    plt.show()
#                 elif sats==False:
#                     disp = ConfusionMatrixDisplay.from_predictions(
#                         targ_te,
#                         np.argmax(label_pred_te, axis=1),
#                         display_labels=["Background", "  Brightening", "Dimming", "Cosmic Ray", "Streak"],
#                         cmap="bone",
#                         normalize=tonorm,
#                         )
#                         # Adjust tick label size
#                     disp.ax_.tick_params(axis="y", labelsize=17)
#                     disp.ax_.tick_params(axis="x", labelsize=13)
            
#                     # Adjust font size of numbers
#                     for text in disp.ax_.texts:  # Access text annotations
#                         text.set_fontsize(16)  # Set desired font size
#                     disp.ax_.set_xlabel("Predicted Label", fontsize=14)  # Adjust X-axis label
#                     disp.ax_.set_ylabel("True Label", fontsize=18)  # Adjust Y-axis label
#                     disp.ax_.set_title(save_name)
            
#                     cbar = disp.im_.colorbar  # Access the colorbar
#                     cbar.ax.tick_params(labelsize=16)  # Adjust tick label size
#                     plt.show()

    
    
    

# our_feat_te = feat_te[bness_te > 0]
# our_bness_te = bness_te[bness_te > 0]
# our_lens_te = lens_te[bness_te > 0]
# our_pred_te_max = np.argmax(label_pred_te[bness_te > 0], axis=1)
# our_targ_te = targ_te[bness_te > 0]
   


def heatmap(save_name, folds=False):
    
    our_feat_te = np.load("/home/3155/saved_loss/our_feat_te_"+save_name+".npy")
    our_bness_te = np.load("/home/3155/saved_loss/our_bness_te_"+save_name+".npy")
    our_lens_te = np.load("/home/3155/saved_loss/our_lens_te_"+save_name+".npy")

    our_targ_te = np.load("/home/3155/saved_loss/our_targ_te_"+save_name+".npy") 
    targ_te = np.load("/home/3155/saved_loss/targ_te_"+save_name+".npy") 
    if folds==False:
        our_pred_te_max = np.load("/home/3155/saved_loss/our_pred_te_max_"+save_name+".npy")
    else:
        #can't determine the max until i average predictions
        foldses = ["0","1","2","3","4"]
        label_pred_tes = []
        for fold in foldses:
            label_pred_tes.append(np.load("/home/3155/saved_loss/label_pred_te_"+save_name+"_fold_"+str(fold)+".npy"))
        label_pred_te = np.average(np.array(label_pred_tes),axis=0)
        # print(label_pred_te.shape)
        targ_te = np.load('/home/3155/saved_loss/targ_te_'+save_name+'.npy')
        mask = np.sum(([targ_te==2],[targ_te==1]), axis=0)[0]
        # print(mask)
        
        # return label_pred_te, mask
        our_pred_te_max = np.argmax(label_pred_te[mask.astype(bool)], axis=1)
        # print("mask above")
        # print(label_pred_te[mask])
        # print("above should be fine")
        # print(our_pred_te_max)
                    
                    
    
    length_values = np.unique(our_lens_te)
    trues = []
    fails = []
    for i in range(len(our_targ_te)):
        if our_targ_te[i] == our_pred_te_max[i]:
            trues.append([our_bness_te[i], our_lens_te[i]])
        else:
            fails.append([our_bness_te[i], our_lens_te[i]])
    fails = np.array(fails)
    trues = np.array(trues)

    # Given nonlinear brightness values
    brightness_values = np.unique(our_bness_te)
    length_values = np.unique(our_lens_te)

    # # Given nonlinear brightness and length values
    # brightness_values = np.array([0.8, 0.9, 0.99, 1.01, 1.1, 2, 5, 10])
    # length_values = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # Example length values

    # Create bins for brightness and length
    brightness_bins = np.concatenate(
        [
            [brightness_values[0] - 0.1],
            (brightness_values[:-1] + brightness_values[1:]) / 2,
            [brightness_values[-1] + 0.1],
        ]
    )

    length_bins = np.concatenate(
        [
            [length_values[0] - 0.5],
            (length_values[:-1] + length_values[1:]) / 2,
            [length_values[-1] + 0.5],
        ]
    )

    # Example data: Replace these with your actual data
    correct_detections = trues  # Array of correctly detected (brightness, length)
    incorrect_detections = fails  # Array of incorrectly detected (brightness, length)

    # Create 2D histogram for correct and total detections
    correct_hist, xedges, yedges = np.histogram2d(
        correct_detections[:, 0],
        correct_detections[:, 1],
        bins=[brightness_bins, length_bins],
    )

    total_hist, _, _ = np.histogram2d(
        np.vstack((correct_detections, incorrect_detections))[:, 0],
        np.vstack((correct_detections, incorrect_detections))[:, 1],
        bins=[brightness_bins, length_bins],
    )

    # Calculate efficiency
    efficiency = correct_hist / total_hist

    # Calculate error (using Poisson approximation)
    error = np.sqrt(efficiency * (1 - efficiency) / total_hist)

    # Plotting the heatmap
    plt.figure(figsize=(12, 12))
    bottom_ticks_values = (
        np.arange(3, len(length_values) + 3) * 10000 / 3080
    )  # * (10/6.15)  # Example: double the length values
    bottom_tick_labels = [f"{val:.1f}" for val in bottom_ticks_values]
    heatmap = sns.heatmap(
        efficiency,
        xticklabels=bottom_tick_labels,
        yticklabels=brightness_values,
        cmap="viridis",
        annot=False,
        vmin=0,
        vmax=1,
    )

    # Setting the bottom axis label
    ax = plt.gca()  # Get the current axis
    ax.set_xlabel("Transient Duration $\sigma$ (milliseconds)", fontsize=18)  # Bottom label
    ax.xaxis.set_label_position("bottom")  # Ensure the label is positioned at the bottom
    ax.xaxis.tick_bottom()  # Ensure the ticks are at the bottom

    # Customizing the bottom ticks
    # plt.xticks(ticks=bottom_ticks_values, labels=bottom_tick_labels, fontsize=12)
    plt.xticks(fontsize=12)

    # Creating custom ticks and label for the top axis with different numbers
    top_tick_labels = length_values
    ax_top = ax.twiny()  # Create a new axis sharing the x-axis
    ax_top.set_xlim(ax.get_xlim())  # Ensure the limits are shared
    ax_top.set_xticks(np.arange(len(length_values)) + 0.5)  # Set tick positions for top
    ax_top.set_xticklabels(top_tick_labels, fontsize=13)  # Set custom labels for top
    ax_top.set_xlabel(
        "Transient Duration $\sigma$ (pixels)", fontsize=18, labelpad=10
    )  # Top label

    # Setting y-axis label and title
    ax.set_ylabel("Transient Multiplicative Amplitude A", fontsize=18)
    plt.title("Transient Detection Efficiency Heatmap", fontsize=20)

    # Customizing tick positions and labels for the y-axis
    plt.yticks(
        ticks=np.arange(len(brightness_values)) + 0.5, labels=brightness_values, fontsize=12
    )

    # Overlaying error bars and adding text annotations
    for i in range(efficiency.shape[0]):
        for j in range(efficiency.shape[1]):
            if total_hist[i, j] > 0:  # Only annotate if there are detections
                # Format the annotation text
                text = f"{efficiency[i, j]:.2f} ± {error[i, j]:.2f}" #{total_hist[i,j]}"#
                # Add the text annotation in the center of each bin
                if efficiency[i, j]<.8:
                    color="white"
                else:
                    color='orange'
                plt.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=11,
                )

                # Remove the black outline (spines) from all sides
    for spine in ax.spines.values():
        spine.set_visible(False)

    for spine in ax_top.spines.values():
        spine.set_visible(False)


    # Adjust tick label size
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=13)


    # Access the colorbar and modify tick font size
    cbar = heatmap.collections[0].colorbar  # Access the colorbar
    cbar.ax.tick_params(labelsize=13)  # Adjust tick label size
    
    
def make_aggregated_transient_detections_df(olddf, threshold=30):
    '''
    this de-duplicates transients that are within threshold, both pos and neg transients
    30 performs the best for thr=.99 
    lower than 30 decreases precision'''
    df = olddf.copy() #just to make sure i dont accidentally modify the old one in place
    newdf = pd.DataFrame()
    for filename, filenamegroup in df.groupby('filename'):  
        for quad, quadgroup in filenamegroup.groupby('quad'):
            for closest_x, closest_xgroup in quadgroup.groupby('closest_x'):
                ignore_in_this_group = []
                for i in closest_xgroup.index:
                    closest_idx = []
                    original_col = closest_xgroup.loc[i,'col_from_nn']
                    original_row = closest_xgroup.loc[i,'row_from_nn']
                    for j in closest_xgroup.index:
                        if j not in ignore_in_this_group:
                            if (np.abs(closest_xgroup.loc[j,'col_from_nn']-original_col)<5)*(np.abs(closest_xgroup.loc[j,'row_from_nn']-original_row)<threshold)*(j not in ignore_in_this_group):
                                # if (closest_xgroup.loc[j,'dist_to_implant']>20) or (closest_xgroup.loc[j,'A_from_implant']==closest_xgroup.loc[i,'A_from_implant']): 
                                #     #either matched to the same one or far away
                                    closest_idx.append(j)
                                    ignore_in_this_group.append(j)
                        # print(closest_idx)
                    if len(closest_idx)>0:
                        new_transient_y = np.average(closest_xgroup.loc[closest_idx, 'row_from_nn'])
                        original_rows = np.array(closest_xgroup.loc[closest_idx, 'row_from_nn'])
                        new_transient_x = np.average(closest_xgroup.loc[closest_idx, 'col_from_nn'])
                        original_cols = np.array(closest_xgroup.loc[closest_idx, 'col_from_nn'])
                        # print(closest_xgroup.loc[closest_idx, 'col_from_nn'])
                        # print(new_transient_x)
                        therow = closest_xgroup.loc[i].copy()
                        detection_sizes = np.array(closest_xgroup.loc[closest_idx, 'detection_size'])
                        therow['row_from_nn']=new_transient_y
                        therow['col_from_nn']=new_transient_x
                        therow['original_cols_from_nn']=original_cols
                        therow['original_rows_from_nn']=original_rows
                        therow['detection_size']=np.sum(closest_xgroup.loc[closest_idx, 'detection_size'])
                        
                        detection_sizes = np.array(closest_xgroup.loc[closest_idx, 'detection_size'])
                        therow['detection_sizes']=detection_sizes
                        if len(closest_idx)>1:
                            modified_flag=True
                            therow['type']='both'
                            
                        else:
                            modified_flag=False
                        therow['modified_flag'] = modified_flag
                        therow['original_types']=np.array(closest_xgroup.loc[closest_idx, 'type'])
                        # therow['dist_to_implant'] = np.sqrt((new_transient_x-closest_xgroup.loc[i, 'col_from_implant'])**2+(new_transient_y-closest_xgroup.loc[i, 'row_from_implant'])**2)
                        # print(therow)
                        newdf = pd.concat([newdf, therow.to_frame().T], ignore_index=True)
                    # newdf.append(therow, ignore_index=True)
                        # print(newdf)
                        # closest_xgroup.drop(index=closest_idx)
                        # print(len(closest_xgroup))
            # break
        # for idx in group.index:
    return newdf

def get_snr_dimmest(tot, aggrows):
    '''
    inputs: tot as in tot, maxcols = get_tot(df,index)
    aggrow is the collection of rows (positions along the streak that were agg), could be len 1
    outputs: 
    snr_local
    snr_global
    noise_local
    noise_global
    local_average_outside: baseline to comparison to height
    height
    FWHM: full width half max
    signal_local
    '''
    newwid = 10
    #check within newwid on each side, if there is more than 
    if int(np.min(aggrows))>10: #fixing iissues when too close to sides
        lowerlim = int(np.min(aggrows))
    else:
        lowerlim = 10
    if int(np.max(aggrows))<3070:
        upperlim = int(np.max(aggrows))
    else:
        upperlim = 3070
    tallest_index = np.argmin(tot[lowerlim - newwid : upperlim + newwid])
    # print("tallest_index, ", tallest_index)
    tallest_index_absolute = tallest_index+(int(np.min(aggrows)) - newwid)
    
    #get bounds for measureing the noise
    leftestleft, rightestleft, leftestright, rightestright = tallest_index_absolute - 200 - newwid,tallest_index_absolute - 100 - newwid, tallest_index_absolute + 100 + newwid,  tallest_index_absolute + 200 + newwid
    bounds = np.array([leftestleft, rightestleft, leftestright, rightestright])
    for i, bound in enumerate(bounds): #adjust if too close to the edge
        if bound < 0:
            bounds[i] = 0
        elif bound > 3079:
            bounds[i] = 3079
    local_outside = np.concatenate((tot[ bounds[0] : bounds[1]], tot[bounds[2] : bounds[3]]))
    local_average_outside = np.average(local_outside)
    tot_subtracted = tot - local_average_outside
    height = tot_subtracted[tallest_index_absolute]
    half_max = height/2
    row = tallest_index_absolute #so the rest can stay the same
    arr_left = tot_subtracted[ : row ][::-1]
    arr_right = tot_subtracted[row : ]
    left_index = next((i for i,v in enumerate(arr_left) if v>half_max), len(arr_left)-1) 
    right_index = next((i for i,v in enumerate(arr_right) if v>half_max), len(arr_right)-1) 
    if right_index==0:
        right_index=1
    #making sure to include the row itself if none found
    # print("indices for size of integral: ", left_index, right_index)
    # print(height)
    # print(half_max)
    # print(local_average_outside)
    window_minus_average = tot_subtracted[row - left_index : row + right_index]
    FWHM = len(window_minus_average)
    # print(window_minus_average)
    integrated_window = np.sum(window_minus_average)
    signal_local = np.sqrt(np.abs(integrated_window))
    if integrated_window > 0: #we take the absolute value to get the signal, but i need to show
        #that it is meaningless if the neg transient is not actually below zero etc
        signal_local = -signal_local
    noise_local = np.std(local_outside)
    noise_global = np.std(np.concatenate((tot[ 100 : row - newwid**2 - 10], tot[row + newwid**2 + 10: -100])))
    if 100 > row - newwid**2:
        noise_global = np.std(tot[row + newwid**2 + 10: -100])
    elif row + newwid**2 > 2979:
        noise_global = np.std(tot[ 100 : row - newwid**2 - 10])
    else:
        noise_global = np.std(np.concatenate((tot[ 100 : row - newwid**2 - 10], tot[row + newwid**2 + 10: -100])))

    snr_local = signal_local/noise_local
    snr_global = signal_local/noise_global
    return snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local, tallest_index_absolute


def get_snr_brightest(tot, aggrows):
    '''
    inputs: tot as in tot, maxcols = get_tot(df,index)
    aggrow is the collection of rows (positions along the streak that were agg), could be len 1
    outputs: 
    snr_local
    snr_global
    noise_local
    noise_global
    local_average_outside: baseline to comparison to height
    height
    FWHM: full width half max
    signal_local
    '''
    newwid = 10
    #check within newwid on each side, if there is more than 
    if int(np.min(aggrows))>10: #fixing iissues when too close to sides
        lowerlim = int(np.min(aggrows))
    else:
        lowerlim = 10
    if int(np.max(aggrows))<3070:
        upperlim = int(np.max(aggrows))
    else:
        upperlim = 3070
    tallest_index = np.argmax(tot[lowerlim - newwid : upperlim + newwid])
    # print("tallest_index, ", tallest_index)
    tallest_index_absolute = tallest_index+(int(np.min(aggrows)) - newwid)
    
    #get bounds for measureing the noise
    leftestleft, rightestleft, leftestright, rightestright = tallest_index_absolute - 200 - newwid,tallest_index_absolute - 100 - newwid, tallest_index_absolute + 100 + newwid,  tallest_index_absolute + 200 + newwid
    bounds = np.array([leftestleft, rightestleft, leftestright, rightestright])
    for i, bound in enumerate(bounds): #adjust if too close to the edge
        if bound < 0:
            bounds[i] = 0
        elif bound > 3079:
            bounds[i] = 3079
    local_outside = np.concatenate((tot[ bounds[0] : bounds[1]], tot[bounds[2] : bounds[3]]))
    local_average_outside = np.average(local_outside)
    tot_subtracted = tot - local_average_outside
    height = tot_subtracted[tallest_index_absolute]
    half_max = height/2
    row = tallest_index_absolute #so the rest can stay the same
    arr_left = tot_subtracted[ : row ][::-1]
    arr_right = tot_subtracted[row : ]
    left_index = next((i for i,v in enumerate(arr_left) if v<half_max), len(arr_left)-1) #there was a +1 on each o fthese
    right_index = next((i for i,v in enumerate(arr_right) if v<half_max), len(arr_right)-1) 
    if right_index==0:
        right_index=1
    #making sure to include the row itself if none found
    # print("indices for size of integral: ", left_index, right_index)
    # print(height)
    # print(half_max)
    # print(local_average_outside)
    window_minus_average = tot_subtracted[row - left_index : row + right_index ] #want to include the rightmost point
    # print(window_minus_average)
    FWHM = len(window_minus_average)
    
    integrated_window = np.sum(window_minus_average)
    signal_local = np.sqrt(np.abs(integrated_window))
    if integrated_window < 0: #we take the absolute value to get the signal, but i need to show
        #that it is meaningless if the neg transient is not actually below zero etc
        signal_local = -signal_local
    noise_local = np.std(local_outside)
    noise_global = np.std(np.concatenate((tot[ 100 : row - newwid**2 - 10], tot[row + newwid**2 + 10: -100])))
    if 100 > row - newwid**2:
        noise_global = np.std(tot[row + newwid**2 + 10: -100])
    elif row + newwid**2 > 2979:
        noise_global = np.std(tot[ 100 : row - newwid**2 - 10])
    else:
        noise_global = np.std(np.concatenate((tot[ 100 : row - newwid**2 - 10], tot[row + newwid**2 + 10: -100])))

    snr_local = signal_local/noise_local
    snr_global = signal_local/noise_global
    return snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local, tallest_index_absolute





def add_snr_both(olddf):
    '''
    add pos and neg integrated snrs
    returns the dataframe with new columns for outputs of
    snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, brightest/dimmest row 
    currently hardcoded half ap = 2
    '''
    wid=10
    df = olddf.copy()
    column_names_brightening = ['snr_local_brightening', 'snr_global_brightening', 'noise_local_brightening', 'noise_global_brightening', 
                'local_average_outside_brightening', 'height_brightening', 'FWHM_brightening', 'signal_local_brightening', 'central_point_brightening']
    column_names_dimming = ['snr_local_dimming', 'snr_global_dimming', 'noise_local_dimming', 'noise_global_dimming', 
                'local_average_outside_dimming', 'height_dimming', 'FWHM_dimming', 'signal_local_dimming', 'central_point_dimming']

    for col in column_names_brightening:
        df[col] = np.zeros(len(df))
    for col in column_names_dimming:
        df[col] = np.zeros(len(df))
    for filename, filenamegroup in df.groupby('filename'):  
        for quad, quadgroup in filenamegroup.groupby('quad'):
            for closest_x, closest_xgroup in quadgroup.groupby('closest_x'): #which light curve it is
                tot, maxcols = get_tot_csv(df, closest_xgroup.index[0]) #will be the same for all so just using the first
                for index in closest_xgroup.index:
                    # row = int(df.loc[index, "row_from_nn"]) + wid
                    aggrows = df.loc[index, "original_rows_from_nn"] + wid
                    #first brightening
                    snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local, tallest_index_absolute = get_snr_brightest(tot, aggrows)                    
                    df.loc[index, column_names_brightening] = [snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local, tallest_index_absolute]
                    #now dimming
                    snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local, tallest_index_absolute = get_snr_dimmest(tot, aggrows)                    
                    df.loc[index, column_names_dimming] = [snr_local, snr_global, noise_local, noise_global, local_average_outside, height, FWHM, signal_local, tallest_index_absolute]
    return df            

