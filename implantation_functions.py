import numpy as np
import matplotlib.pyplot as plt


def gaussian_event(ys, M, sig, A, original_values):
    """ys should be the array of y values
    #m is the center y value
    #sig is the standard deviation
    #original_values is the original brightness (for each y value)
    #A is the maximum brightening. A-1 is in the code bc of that additional 1 later
    #return original_values * ((A-1) * (np.exp(-np.power(ys - M, 2.) / (2 * np.power(sig, 2.)))) + 1)
    """
    return original_values * (
        (A - 1) * (np.exp(-np.power(ys - M, 2.0) / (2 * np.power(sig, 2.0)))) + 1
    )


def get_intensities_pos(total_desired, seed):
    """
    the positive intensities
    o	For the center of the transient, choose to have at least half the transient in the image
    (randint between -5 and 26 will be the center of the gaussian for size 10 events)
    o	intensities of (multiplicative factors) 7, 8, 9, 1.5, 1.3, 1.1, 1.05, 1.01
    """
    np.random.seed(
        seed
    )  # so that i can keep the same random - last time it was seed 21. before that it was 20

    # for the center of the transient - want at least half of the transient in the image
    center_values = np.random.randint(15, high=3065, size=58500, dtype=int)
    # adjusted this to be the whole image
    evensplit = (
        total_desired // 11 + 1
    )  # to make sure there is enough i will put one extra of each
    intensities_8 = np.full(int(evensplit), 8)
    intensities_7 = np.full(int(evensplit), 1.5)
    intensities_105 = np.full(int(evensplit), 1.3)
    intensities_101 = np.full(int(evensplit), 1.1)
    intensities_102 = np.full(int(evensplit), 1.05)
    intensities_103 = np.full(int(evensplit), 1.01)
    intensities_2 = np.full(int(evensplit), 2)
    intensities_3 = np.full(int(evensplit), 3)
    intensities_4 = np.full(int(evensplit), 4)
    intensities_6 = np.full(int(evensplit), 6)
    intensities_10 = np.full(int(evensplit), 10)

    intensities = np.concatenate(
        (
            intensities_2,
            intensities_3,
            intensities_4,
            intensities_6,
            intensities_10,
            intensities_101,
            intensities_102,
            intensities_103,
            intensities_7,
            intensities_8,
            intensities_105,
        )
    )

    np.random.shuffle(intensities)  # this shuffles the array in place
    return intensities


def get_intensities_neg(total_desired, seed):
    """
    the negative intensities
    o	For the center of the transient, choose to have at least half the transient in the image
    (randint between -5 and 26 will be the center of the gaussian for size 10 events)
    o	intensities of (multiplicative factors) 7, 8, 9, 1.5, 1.3, 1.1, 1.05, 1.01
    """
    np.random.seed(
        seed
    )  # so that i can keep the same random - last time it was seed 21. before that it was 20

    # for the center of the transient - want at least half of the transient in the image
    center_values = np.random.randint(15, high=3065, size=58500, dtype=int)
    # adjusted this to be the whole image
    evensplit = (
        total_desired // 5 + 1
    )  # to make sure there is enough i will put one extra of each
    intensities_8 = np.full(int(evensplit), 0.99)
    intensities_7 = np.full(int(evensplit), 0.95)
    intensities_105 = np.full(int(evensplit), 0.9)
    intensities_101 = np.full(int(evensplit), 0.85)
    intensities_102 = np.full(int(evensplit), 0.8)
    intensities_103 = np.full(int(evensplit), 0.995)
    # intensities_2 = np.full(int(evensplit), 2)
    # intensities_3 = np.full(int(evensplit), 3)
    # intensities_4 = np.full(int(evensplit), 4)
    # intensities_6 = np.full(int(evensplit), 6)
    # intensities_10 = np.full(int(evensplit), 10)

    intensities = np.concatenate(
        (  # intensities_2, intensities_3, intensities_4,
            # intensities_6, intensities_10,
            intensities_102,
            intensities_101,
            intensities_103,
            intensities_7,
            intensities_8,
            intensities_105,
        )
    )

    np.random.shuffle(intensities)  # this shuffles the array in place
    return intensities


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


def implant_transient_full_image(image, streak_x_val, center_value_y, intensity, sigma):
    """
    this function takes an image and implants a gaussian brightening event in the given columns (in a streak).
    inputs -
    image: an array of brightness values
    streak_x_val: location of the streak - x coordinate
    center_value_y: the y value to center the gaussian at
    sigma: the sigma of the gaussian
    intensity: the maximum brightening of the gaussian
    returns:  gaussianed_image, normalized array with brightening in the streak
    """

    # getting background
    background = get_image_background(image)

    # subtracting background
    imagenobackground = image - np.full(np.shape(image), background)

    # choosing which columns are part of the streak (to implant transient in)
    # careful: there could be two streaks in the image, i want the one that is the most centered
    # for now we're just taking the columns around the streak
    mid = streak_x_val  # the middle column - truncated. so the lower one if even number of columns

    averages = np.average(
        imagenobackground[center_value_y - 10 : center_value_y + 10], axis=0
    )
    # there should be some criteria for the gradient perhaps
    # failing that, just that it's bigger than .05
    column_indices = []
    # indices_to_check = []

    # if mid < 3072-4:
    #     indices_to_check = [mid-4, mid-3, mid-2, mid-1, mid, mid+1, mid+2, mid+3]
    # for index in indices_to_check:
    #     if averages[index]>.01:
    #         column_indices.append(index)

    maxcol = np.argmax(averages[streak_x_val - 3 : streak_x_val + 3]) + streak_x_val - 3
    maxval = averages[maxcol]
    column_indices.append(maxcol)
    column_indices.append(maxcol + 1)
    val = averages[maxcol + 1]
    # print(maxcol)
    for i in range(2, 6):
        index = maxcol + i
        # print(index)
        newval = averages[index]
        if (
            newval > val or newval < 0.1 * maxval
        ):  # if the value is bigger than the last one or if it's less than 10% of its max, we're done
            break
        val = newval
        column_indices.append(index)
    column_indices.append(maxcol - 1)
    val = averages[maxcol - 1]
    for i in range(2, 6):
        index = maxcol - i
        newval = averages[index]
        # print(index)
        if (
            newval > val or newval < 0.1 * maxval
        ):  # if the value is bigger than the last one or if it's less than 10% of its max, we're done
            break
        val = newval
        column_indices.append(index)
    print(column_indices)

    # now gaussianning all relevant columns

    gaussianed_image = (
        imagenobackground.copy()
    )  # stamp array to modify with the gaussian brightening

    for i in column_indices:
        gaussianed = gaussian_event(
            ys=np.linspace(0, image.shape[0] - 1, image.shape[0]),
            M=center_value_y,
            sig=sigma,
            A=intensity,
            original_values=imagenobackground[:, i],
        )
        gaussianed_image[:, i] = gaussianed

    # adding back background
    gaussianed_image = gaussianed_image + np.full(np.shape(image), background)

    # should renormalize it
    norm_gi = (gaussianed_image - np.min(gaussianed_image)) / (
        np.max(gaussianed_image) - np.min(gaussianed_image)
    )

    return norm_gi


def implant_10_transients(
    image, sigmas, intensities, center_values, counter, plot=False
):
    """
    like it says on the tin

    """
    intensity_counter = counter
    y_counter = counter
    transient_info = []  # empty list for adding info for implanted transients

    # short list of where brightest streaks are
    averages = np.average(image[:3040, :3040], axis=0)
    sorted = np.argsort(averages)[-20:]
    doublesorted = np.sort(sorted)
    shortlist = []
    for i, el in enumerate(doublesorted):
        ihatenumbers = np.hstack((doublesorted, [0]))
        if ihatenumbers[i + 1] != el + 1:
            shortlist.append(el)

    implanted = image
    for streak_x_val in shortlist[:10]:
        center_value_y = center_values[y_counter]
        intensity = intensities[intensity_counter]
        sigma = sigmas[intensity_counter]
        y_counter += 1
        intensity_counter += 1
        implanted = implant_transient_full_image(
            implanted, streak_x_val, center_value_y, intensity, sigma
        )
        # print("intensity "+str(intensity))
        # print("xval "+str(streak_x_val))
        if plot == True:
            show_streak(implanted, streak_x_val, center_value_y)
        trans_xcoord_ycoord_intensity_sigma = (
            streak_x_val,
            center_value_y,
            intensity,
            sigma,
        )
        print("trans xcoord etc: ", trans_xcoord_ycoord_intensity_sigma)
        transient_info.append(trans_xcoord_ycoord_intensity_sigma)
    return implanted, transient_info


def get_stamp(image, streak_x_val, center_value_y, plot=False):
    streak_x_val = int(streak_x_val)
    center_value_y = int(center_value_y)
    # plt.imshow(np.sqrt(image), aspect=.05)
    # plt.xlim(streak_x_val-10, streak_x_val+10)
    # #plt.ylim(center_value_y-20, center_value_y+20)
    # plt.colorbar()
    # plt.show()
    stamp = image[
        center_value_y - 10 : center_value_y + 10, streak_x_val - 10 : streak_x_val + 10
    ]
    # plt.imshow(np.sqrt(stamp))
    # # plt.xlim(streak_x_val-20, streak_x_val+20)
    # # plt.ylim(center_value_y-20, center_value_y+20)
    # # plt.colorbar()
    # plt.show()
    if plot == True:
        plt.imshow(np.sqrt(stamp))
        plt.show()
    return stamp


def get_10_trans_stamps(image, sigmas, intensities, center_values, counter, plot=False):
    """
    mmmmmm thinking

    """
    intensity_counter = counter
    y_counter = counter
    stamp_info = []  # empty list for adding info for implanted transients

    # short list of where brightest streaks are
    averages = np.average(image, axis=0)
    sorted = np.argsort(averages)[-20:]
    doublesorted = np.sort(sorted)
    shortlist = []
    for i, el in enumerate(doublesorted):
        ihatenumbers = np.hstack((doublesorted, [0]))
        if ihatenumbers[i + 1] != el + 1 and el > 10 and el < (len(averages) - 10):
            shortlist.append(el)

    implanted = image
    stamps_implanted = []
    for streak_x_val in shortlist[:10]:
        center_value_y = center_values[y_counter]
        intensity = intensities[intensity_counter]
        sigma = sigmas[intensity_counter]
        y_counter += 1
        intensity_counter += 1
        implanted = implant_transient_full_image(
            implanted, streak_x_val, center_value_y, intensity, sigma
        )
        # print("intensity "+str(intensity))
        # print("xval "+str(streak_x_val))
        if plot == True:
            # show_streak(implanted, streak_x_val, center_value_y)
            get_stamp(implanted, streak_x_val, center_value_y, plot=True)
        intensity_sigma = (intensity, sigma, streak_x_val,center_value_y)
        print("stamp info: ", intensity_sigma)
        stamp_info.append(intensity_sigma)
        stamp = get_stamp(implanted, streak_x_val, center_value_y, plot=False)
        stamps_implanted.append(stamp)
    return stamps_implanted, stamp_info


def get_10_streak_stamps(image, center_values, counter, plot=False):
    """
    mmmmmm thinking

    """
    intensity_counter = counter
    y_counter = counter

    # short list of where brightest streaks are
    averages = np.average(image, axis=0)
    sorted = np.argsort(averages)[-20:]
    doublesorted = np.sort(sorted)
    shortlist = []
    for i, el in enumerate(doublesorted):
        ihatenumbers = np.hstack((doublesorted, [0]))
        if ihatenumbers[i + 1] != el + 1 and el > 10 and el < (len(averages) - 10):
            shortlist.append(el)

    implanted = image
    stamps_implanted = []
    for streak_x_val in shortlist[:10]:
        center_value_y = center_values[y_counter]
        y_counter += 1
        intensity_counter += 1
        # implanted = implant_transient_full_image(implanted, streak_x_val, center_value_y, intensity, sigma)
        # # print("intensity "+str(intensity))
        # # print("xval "+str(streak_x_val))
        if plot == True:
            # show_streak(implanted, streak_x_val, center_value_y)
            get_stamp(implanted, streak_x_val, center_value_y, plot=True)
        stamp = get_stamp(implanted, streak_x_val, center_value_y, plot=False)
        stamps_implanted.append(stamp)
    return stamps_implanted


def intermediate_values(shortlist):
    intermediate_values = []
    for i in range(len(shortlist) - 1):
        if shortlist[i] + 25 < shortlist[i + 1]:
            intermed = int((shortlist[i] + shortlist[i + 1]) / 2)
            intermediate_values.append(intermed)
    return intermediate_values


def get_10_background_stamps(image, center_values, counter, plot=False):
    """
    mmmmmm thinking

    """
    intensity_counter = counter
    y_counter = counter

    # short list of where brightest streaks are
    averages = np.average(image, axis=0)
    sorted = np.argsort(averages)[-70:]
    doublesorted = np.sort(sorted)
    shortlist = []
    for i, el in enumerate(doublesorted):
        ihatenumbers = np.hstack((doublesorted, [0]))
        if ihatenumbers[i + 1] != el + 1 and el > 10 and el < (len(averages) - 10):
            shortlist.append(el)

    int_vals = intermediate_values(shortlist)

    implanted = image
    stamps_bg = []
    for bg_x_val in int_vals:
        center_value_y = center_values[y_counter]
        y_counter += 1
        intensity_counter += 1
        if plot == True:
            get_stamp(implanted, bg_x_val, center_value_y, plot=True)
        stamp = get_stamp(implanted, bg_x_val, center_value_y, plot=False)
        stamps_bg.append(stamp)
    return stamps_bg
