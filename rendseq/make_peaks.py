# -*- coding: utf-8 -*-
"""Take normalized raw data find the peaks in it."""

import argparse
import sys
import warnings
from os.path import abspath

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from rendseq.file_funcs import make_new_dir, open_wig, write_wig


def _populate_trans_mat(z_scores, peak_center, spread, trans_m, states):
    """Calculate the Vertibi Algorithm transition matrix.

    Parameters
    ----------
        -z_scores (2xn array): - required: first column is position (ie bp
            location) second column is a modified z_score for that position.
        -peak_center (float): the mean of the emission probability distribution
            for the peak state.
        -spread (float): the standard deviation of the peak emmission
            distribution.
        -trans_m (matrix): the transition probabilities between states.
        -states (matrix): how internal and peak are represented in the wig file

    Raises
    ------
        -ValueError if the HMM parameters are unable to find any likely paths.
            This is at risk of happening if the emission probabilities for
            either the peak state or the internal state look very different from
            the true distribution, or the parameters selected only allow a very
            unlikely path - for example if p_to_p = 1 and you try to force the
            system to stay in the peak state.
    """
    print("Calculating Transition Matrix")
    trans_1 = np.zeros([len(states), len(z_scores)])
    trans_2 = np.zeros([len(states), len(z_scores)]).astype(int)
    trans_1[:, 0] = 1

    # emission probabilities matrix (2 x n)
    probs = np.zeros(shape=(len(states), len(z_scores)))
    probs[0, :] = norm.pdf(
        z_scores[:, 1]
    )  # note zscores[:,1] is where all the zscores are
    probs[1, :] = norm.pdf(z_scores[:, 1], peak_center, spread)

    # Vertibi Algorithm:
    for i in range(1, len(z_scores)):
        # we use log probabilities for computational reasons. -Inf means 0 probability
        paths = np.zeros((len(states), len(states)))
        paths += np.expand_dims(
            trans_1[:, i - 1], axis=1
        )  # adds log(trans_1_sub) column-wise
        paths += np.log(trans_m)  # adds log(trans_m) element-wise
        paths += np.expand_dims(
            np.log(probs[:, i]), axis=0
        )  # adds log(probs_sub) row-wise
        trans_2[:, i] = np.argmax(paths, axis=0)
        trans_1[:, i] = np.max(paths, axis=0)

    if np.any(np.isinf(trans_1[:, -1])):
        raise ValueError(
            "".join(
                [
                    "Parameters Provided are too far from the true",
                    "distribution for a valid path to be found.  Please ",
                    "change your paramters for hmm_peaks and try again.",
                ]
            )
        )
    return trans_1, trans_2


def hmm_peaks(z_scores, i_to_p=1 / 1000, p_to_p=1 / 1.5, peak_center=10, spread=2):
    """Fit peaks to the provided z_scores data set using the vertibi algorithm.

    Parameters
    ----------
        -z_scores (2xn array): - required: first column is position (ie bp
            location) second column is a modified z_score for that position.
        -i_to_p (float): value should be between zero and 1, represents
            probability of transitioning from inernal state to peak state. The
            default value is 1/2000, based on asseumption of geometrically
            distributed transcript lengths with mean length 1000. Should be a
            robust parameter.
        -p_to_p (float): The probability of a peak to peak transition.  Default
            1/1.5.
        -peak_center (float): the mean of the emission probability distribution
            for the peak state.
        -spread (float): the standard deviation of the peak emmission
            distribution.

    Returns
    -------
        -peaks: a 2xn array with the first column being position and the second
            column being a peak assignment.
    """
    print("Finding Peaks")
    max_z = peak_center + spread
    # cap the z scores at peak_center + spread to eliminate very large zscores.
    trim_zscores = z_scores[:, :]
    trim_zscores[:, 1] = np.where(z_scores[:, 1] > max_z, max_z, z_scores[:, 1])
    trans_m = np.asarray(
        [[(1 - i_to_p), (i_to_p)], [p_to_p, (1 - p_to_p)]]
    )  # transition probability
    states = [1, 100]  # how internal and peak are represented in the wig file
    trans_1, trans_2 = _populate_trans_mat(
        trim_zscores, peak_center, spread, trans_m, states
    )
    peaks = np.zeros([len(z_scores), 2])
    peaks[:, 0] = trim_zscores[:, 0]
    # Now we trace backwards and find the most likely path:
    max_inds = np.zeros([len(peaks)]).astype(int)
    max_inds[-1] = int(np.argmax(trans_1[:, len(trans_1)]))
    peaks[-1, 1] = states[max_inds[-1]]
    for index in reversed(list(range(1, len(peaks)))):
        max_inds[index - 1] = trans_2[max_inds[index], index]
        peaks[index - 1, 1] = states[max_inds[index - 1]]
    print(f"Found {sum(peaks[:,1] == states[1])} Peaks")
    return peaks


def _make_kink_fig(save_file, seen, exp, pnts, thresh):
    """Create a figure comparing the obs vs exp z score distributions.

    Parameters
    ----------
        - save_fig (str) - the name of the file to save the plot to.
        - seen (1xn array) - the obs number of positions with a given z score or greater
        - exp (1xn array) - the exp number of positions with a given z score or greater
        - pnts (1xn array) - the z score values/x axis of the plot.
        - thresh (int) - the threshold value which was ultimately selected.
    """
    plt.plot(pnts, seen, label="Observed")
    plt.plot(pnts, exp, label="Expected")
    plt.plot([thresh, thresh], [max(exp) * 10, min(exp) / 10], label="Threshold")
    plt.yscale("log")
    plt.ylabel("Number of Positions with Z score Greater than or equal to")
    plt.xlabel("Z score")
    plt.legend()
    plt.savefig(save_file)


def _calc_thresh(z_scores, method, kink_img="./kink.png"):
    """Calculate a threshold for z-scores file using the method provided.

    Parameters
    ----------
        - z_scores (2xn array): the calculated z scores, where the first column
            represents the nt position and the second represents a z score.
        - method (string): the name of the threshold calculating method to use.

    Returns
    -------
        - threshold (float): the calculated threshold.
    """
    methods = ["expected_val", "kink"]
    thresh = 15
    if method == "expected_val":  # threshold such that num peaks exp < 1.
        p_val = 1 / len(z_scores)  # note this method is dependent on genome size
        thresh = round(norm.ppf(1 - p_val), 1)
    elif method == "kink":  # where the num z_scores exceeds exp num by 10000x
        factor_exceed = 10000
        pnts = np.arange(0, 20, 0.1)
        seen = [0 for i in range(len(pnts))]
        exp = [0 for i in range(len(pnts))]
        thresh = -1
        for ind, point in enumerate(pnts):
            seen[ind] = np.sum(z_scores[:, 1] > point)
            exp[ind] = (1 - norm.cdf(point)) * len(z_scores)
            if seen[ind] >= factor_exceed * exp[ind] and thresh == -1:
                thresh = point

        _make_kink_fig(kink_img, seen, exp, pnts, thresh)

    else:
        warnings.warn(
            "\n".join(
                [
                    f"The method selected ({method}) is not supported.",
                    f"Please select one from {methods}.",
                    f"Defaulting to threshold of {thresh}.",
                ]
            )
        )
    return thresh


def thresh_peaks(z_scores, thresh=None, method="kink", kink_image="./kink.png"):
    """Find peaks by calling z-scores above a threshold as a peak.

    Parameters
    ----------
        - z_scores - a 2xn array of nt positions and zscores at that pos.
        - thresh - the threshold value to use.  If none is provided it will be
            automatically calculated.
        - method - the method to use to automatically calculate the z score
            if none is provided.  Default method is "kink"
    """
    if thresh is None:
        thresh = _calc_thresh(z_scores, method, kink_image)
    peaks = np.zeros([len(z_scores), 2])
    peaks[:, 0] = z_scores[:, 0]
    peaks[:, 1] = (z_scores[:, 1] > thresh).astype(int)
    return peaks


def parse_args_make_peaks(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Can run from the\
                                        commmand line.  Please pass a \
                                        zscore file and select a method \
                                        for peak fitting."
    )
    parser.add_argument("filename", help="Location of the zscore file")
    parser.add_argument(
        "method",
        help='User must pass the desired peak\
                                        fitting method.  Choose "thesh" \
                                        or "hmm"',
    )
    parser.add_argument(
        "--save_file",
        help="Save the z_scores file as a new\
                                        wig file in addition to returning the\
                                        z_scores.  Default = True",
        default=True,
    )
    return parser.parse_args(args)


def main_make_peaks():
    """Run the main peak making from command line."""
    args = parse_args_make_peaks(sys.argv[1:])
    filename = args.filename
    z_scores, chrom = open_wig(filename)
    if args.method == "thresh":
        print(f"Using the thresholding method to find peaks for {filename}")
        peaks = thresh_peaks(z_scores)
    elif args.method == "hmm":
        print(f"Using the hmm method to find peaks for {filename}")
        peaks = hmm_peaks(z_scores)
    else:
        raise ValueError("{args.method} is not a valid peak finding method, see --help")
    if args.save_file:
        filename = abspath(filename).replace("\\", "/")
        file_loc = filename[: filename.rfind("/")]
        peak_dir = make_new_dir([file_loc, "/Peaks/"])
        file_start = filename[filename.rfind("/") + 1 : filename.rfind(".wig")]
        peak_file = "".join([peak_dir, file_start, "_peaks.wig"])
        write_wig(peaks, peak_file, chrom)
        print(f"Wrote peaks to {peak_file}")


if __name__ == "__main__":
    main_make_peaks()
