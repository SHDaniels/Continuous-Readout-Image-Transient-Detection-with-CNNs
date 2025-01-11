import numpy as np
from scipy import optimize
import review_findings_functions as rff


def gaussian_event_streak_val(ys, M, sig, A, streak_val):
    """ys should be the array of y values
    #m is the center y value
    #sig is the standard deviation
    #original_values is the original brightness (for each y value)
    #A is the maximum brightening. A-1 is in the code bc of that additional 1 later
    #return original_values * ((A-1) * (np.exp(-np.power(ys - M, 2.) / (2 * np.power(sig, 2.)))) + 1)
    """
    original_values = np.full(len(ys), streak_val)
    return original_values * (
        (A - 1) * (np.exp(-np.power(ys - M, 2.0) / (2 * np.power(sig, 2.0)))) + 1
    )


# Define the objective function
def function_to_minimize(params, x, ys):
    # Unpack parameters
    M, sig, A, streak_val = params

    # Compute the model's output
    model_output = gaussian_event_streak_val(ys, M, sig, A, streak_val)

    # Calculate the residuals between x and model_output
    residuals = np.abs(x - model_output)

    # Return an aggregated value (e.g., sum of residuals)
    return np.sum(residuals)  # this is L1, could do difference of squares


def fit_params(M_guess, sig_guess, A_guess, streak_val_guess, tot):
    """
    returns:
    print("Optimized M:", optimized_M)
    print("Optimized sig:", optimized_sig)
    print("Optimized A:", optimized_A)
    print("Optimized original_values:", optimized_streak_val)
    print("Minimum value:", result.fun)
    """
    # Fixed arrays for x and ys
    x = tot[M_guess - 60 : M_guess + 61]
    ys = np.linspace(M_guess - 60, M_guess + 60, 121)

    # Initial guesses for M, sig, A, and original_values
    initial_guess = [
        M_guess,
        sig_guess,
        A_guess,
        streak_val_guess,
    ]  # Replace with your actual initial guesses

    # Minimization
    result = optimize.minimize(function_to_minimize, initial_guess, args=(x, ys))

    # Extract optimized parameters
    optimized_M = result.x[0]
    optimized_sig = result.x[1]
    optimized_A = result.x[2]
    optimized_streak_val = result.x[3]
    min_val = result.fun
    return optimized_M, optimized_sig, optimized_A, optimized_streak_val, min_val


def fit_df(
    df,
    form="npy",
    fpath="/lustre/lrspec/ZTF_data/training_aug24/full_image_implantations/images/",
):
    dumbcounter = 0
    optimized_M_arr = np.zeros(len(df))
    optimized_sig_arr = np.zeros(len(df))
    optimized_A_arr = np.zeros(len(df))
    optimized_streak_val_arr = np.zeros(len(df))
    min_val_arr = np.zeros(len(df))

    for i, index in enumerate(df.index):
        if form == "npy":
            tot, maxcols = get_tot_npy(
                df, index, half_aperture=2, tracking=True, fpath=fpath
            )
        else:
            tot, maxcols = rff.get_tot(
                df, index, half_aperture=2, tracking=True, fpath=fpath
            )
        sig_guess = np.sqrt(df.loc[index]["detection_size"])
        M_guess = int(df.loc[index]["row_from_nn"]) + 10

        if M_guess - 60 < 0 or M_guess + 61 > len(tot):
            df.drop(index, inplace=True)
            dumbcounter += 1
            continue

        streak_val_guess = np.mean(
            np.append(
                tot[M_guess - 60 : M_guess - 10], tot[M_guess + 10 : M_guess + 61]
            )
        )
        A_guess = tot[int(M_guess)] / streak_val_guess
        (
            optimized_M,
            optimized_sig,
            optimized_A,
            optimized_streak_val,
            min_val,
        ) = fit_params(M_guess, sig_guess, A_guess, streak_val_guess, tot)
        print(M_guess, sig_guess, A_guess, streak_val_guess)
        print(optimized_M, optimized_sig, optimized_A, optimized_streak_val, min_val)

        optimized_M_arr[i - dumbcounter] = optimized_M
        optimized_sig_arr[i - dumbcounter] = optimized_sig
        optimized_A_arr[i - dumbcounter] = optimized_A
        optimized_streak_val_arr[i - dumbcounter] = optimized_streak_val
        min_val_arr[i - dumbcounter] = min_val
    df["optimized_M"] = optimized_M_arr[: len(df)]
    df["optimized_sig"] = optimized_sig_arr[: len(df)]
    df["optimized_A"] = optimized_A_arr[: len(df)]
    df["optimized_streak_val"] = optimized_streak_val_arr[: len(df)]
    df["min_val"] = min_val_arr[: len(df)]

    return df


def get_tot_npy(
    df,
    index,
    half_aperture=2,
    tracking=True,
    fpath="/lustre/lrspec/ZTF_data/science_extractions/to_extract/1020_minus_extras/",
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
    fname = df.loc[index]["filename"] + "_1_implanted.npy"
    row = (
        int(df.loc[index]["row_from_nn"]) + 10
    )  # we add 10 here bc that's the cnn offset of returned values
    col = int(df.loc[index]["col_from_nn"]) + 10
    quad = int(df.loc[index]["quadrant"])
    column = int(col)

    path = fpath + fname
    # print(path)
    file1_array = np.load(path)
    # norm = (file1_array-np.min(file1_array))/(np.max(file1_array)-np.min(file1_array)) #norm
    image = file1_array

    background = rff.get_image_background(image)

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
