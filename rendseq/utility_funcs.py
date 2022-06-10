# -*- coding: utf-8 -*-
"""Provide general utility functions used in various rendseq analyses."""

import numpy as np


def find_peak_locs(peaks, new_data):
    """
    Find the indices where there are peaks.

    Parameters
    ----------
        - peaks: a 2xn array with the first column being position and the second
            column being a peak assignment.

    Returns
    -------
        - indices: an array with all the indices where peaks can be found.
    """
    peak_value = max(peaks[:, 1])
    peak_locs = peaks[list(np.where(peaks[:, 1] == peak_value)[0]), 0]
    indices = [0 for i in range(len(peak_locs))]
    for p_ind, bp_val in enumerate(peak_locs):
        indices[p_ind] = np.where(new_data[:, 0] == bp_val)[0][0]
    return indices


def zero_padding(data):
    """
    Zero-pad the reads array.

    Add zero values to genomic positions missing from the reads data because
    there were no transcript ends mapped to that position.

    Parameters
    ----------
        - data: a 2xn array containing raw rendseq reads

    Returns
    -------
        - zero_padded_wig: a 2xm array containing zero-padded reads
    """
    print("Zero-padding the reads...")
    ind_start = int(data[0, 0])
    ind_end = int(data[-1, 0])
    indexes = [i for i in range(ind_start, ind_end + 1)]
    values = [0 for i in range(len(indexes))]
    cur_ind = 0
    for ind, bp in enumerate(indexes):
        while data[cur_ind, 0] < bp:
            cur_ind += 1
        if data[cur_ind, 0] == bp:
            values[ind] = data[cur_ind, 1]
        else:
            values[ind] = 0
    zero_padded_wig = np.zeros([len(indexes), 2])
    zero_padded_wig[:, 0] = indexes
    zero_padded_wig[:, 1] = values
    print("Done zero-padding.")
    return zero_padded_wig


def smooth_peaks(peaks_arr, raw_arr):
    """
    Create a "smoothed" or shaved version of the peaks array.

    Take all the peaks called via the peak calling script and
    replace them with the mean of reads after it.

    Parameters
    ----------
        - peaks_arr: a 2xn array with the first column being position and the second
            column being a peak assignment.
        - raw_arr: a 2xn array containing raw rendseq reads

    Returns
    -------
        - smoothed: a 2xn array containing the shaved peaks array
    """
    print("Shaving peaks...")
    smoothed = raw_arr.copy()
    indices = find_peak_locs(peaks_arr, smoothed)
    for p_ind in indices:
        gap = 5
        mean = np.mean(smoothed[p_ind - 3 * gap : p_ind - gap, 1])
        std = np.std(raw_arr[p_ind - 3 * gap : p_ind - gap, 1])
        smoothed[p_ind - gap : p_ind + gap + 1, 1] = abs(
            np.random.normal(mean, std, 2 * gap + 1)
        )
    print("Done shaving peaks.")
    return smoothed
