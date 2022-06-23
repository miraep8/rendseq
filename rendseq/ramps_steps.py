# -*- coding: utf-8 -*-
"""
Find ramps and steps in Rendseq data.

Classes:

    ReadsData
    StretchData

Functions:
    arr_inrange(np.array, np.array, np.array) -> np.array
    output_to_wig(StretchData, StretchData, string, string)
"""

from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats

import rendseq.file_funcs as ff


class ReadsData:
    """
    A class to represent reads data with further processing.

    Provides methods to compute cumulative sum, blurred slope,
    and the derivative of the blurred slope.

    Attributes
    ----------
        - reads: a 2xn array of zero-padded read data
        - dat_idx: a 2xn array of indices, from 0 to len(reads)
        - csum: a length n array of the cumulative sum
        - blurred_data: a length n array of the blurred data
            i.e. the blurred slope of the cumulative sum
        - blurred_deriv: a length n array of the derivative after blurring
            i.e. the derivative of the blurred data


    Methods
    -------
        - process_data(window_width = 400, blur_width = 100):
            Calculates a cumulative sum, blurred data, and derivative.

        - __blurred_slope__(idx, window_width, blur_width):
            Helper method for process_data.

        - __find_window_edges__(center, half_window, blur_width, edge):
            Helper method for blurred_slope.
    """

    def __init__(self, reads):
        """Construct a ReadsData object using a 2nx array of reads."""
        self.reads = reads
        self.dat_idx = np.arange(len(self.reads))

    def __blurred_slope__(self, idx, window_width, blur_width):
        """
        Perform the blurred slope calculation when vectorized.

        Helper method for process_data.
        """
        if idx % 500000 == 0:
            print("index: ", idx, ", time: ", datetime.now())

        # find edges
        half_window = int(1 / 2 * window_width)
        half_blur = int(1 / 2 * blur_width)
        left = self.__find_window_edges__(idx, half_window, half_blur, edge="left")
        right = self.__find_window_edges__(idx, half_window, half_blur, edge="right")

        # if edge indices are invalid, return np.nan as blurry slope
        if left < 0:  # if left is smaller than first possible index
            return np.nan
        if right > len(self.csum) - 1:  # if right is larger than last possible index
            return np.nan

        # take the slope
        arr = self.csum[
            left:right
        ]  # subset of array representing the cumulative reads to take a slope over
        upper_mean = arr[-(2 * half_blur) :].mean()  # the first 2*blur_width reads
        lower_mean = arr[: (2 * half_blur)].mean()  # the last 2*blur_width reads
        slope = (upper_mean - lower_mean) / (2 * half_window)

        return slope

    def __find_window_edges__(self, center, half_window, half_blur, edge):
        """
        Find the left or right edge of the window to take the blured slope over.

        Helper method for blurred_slope.
        """
        if edge == "left":
            return int(center - half_window - half_blur)
        if edge == "right":
            return int(center + half_window + half_blur)
        else:
            raise ValueError("edge must be 'left' or 'right' ")

    def process_data(self, window_width=400, blur_width=200):
        """
        Calculate a cumulative sum, blurred data, and derivative.

        Updates the corresponding attributes.
        """
        # cumulative sum
        self.csum = np.cumsum(self.reads[:, 1])

        # blurred slope of cumulative sum
        print("Calculating blurred slope... This can take a few minutes")
        vblurred_slope = np.vectorize(self.__blurred_slope__)
        self.blurred_data = vblurred_slope(
            self.dat_idx, window_width=window_width, blur_width=blur_width
        )
        print("Done calculating blurred slope")

        # derivative of blurred slope
        self.blurred_deriv = np.gradient(self.blurred_data, self.reads[:, 0])


class StretchData:
    """
    A class to represent stretches of positive or negative slope.

    Provides methods to find contiguous stretches of increasing or
    decreasing read density, filter them based on length, and output results

    Attributes
    ----------
        - reads: a 2xn array of zero-padded read data
        - deriv: a length n array of the derivative after blurring
            i.e. the derivative of the blurred data
        - kind: str, either 'positive' or 'negative'
            Refelects whether stretches of positive slope (increasing)
            or negative slope (decreasing) are being detected
        - search_idx: array of indices of `deriv` where deriv shifts signs
        - search_starts: array of indices of `search_idx` where stretches of
            positive/negative (depending on `kind`) values of deriv begin
        - search_sizes: array of lengths of positive/negative stretches
        - starts: array of indices of `search_idx` where sufficiently long
            (>min_gap) stretches start
        - sizes: array of lengths of sufficiently long positive/negative stretches
        - start_datidx: array of indices in `deriv` where sufficiently long
           stretches start
        - start_genomidx: array of genomic positions where sufficiently long
           stretches start
        - end_genomidx: array of genomic positions where sufficiently long stretches end

    Methods
    -------
        - stretch_search():
            Finds all contiguous stretches of positive or negative derivative,
            depending on `kind`.
        - filter_strenth_length(min_gap):
            Filters out stretches that are smaller than the `min_gap` cutoff.
        - ttest_stretches(gap_len = 50, window_len = 100, pval_cutoff = 1):
            Applies a t-test to the read density before and after each stretch.
        - output_csv(filename, sort_length = True):
            Creates a pandas dataframe of sufficiently long stretches, with genomic
            position and stretch length. Sorts by stretch length (from long to short)
            by default. Outputs the dataframe as a .csv file.

    """

    def __init__(self, readsobj, kind):
        """
        Construct a StretchData object.

        Parameters
        ----------
        readsobj: a ReadsData object with a defined `blurred_deriv` attribute.
        kind: either "positive" (increasing stretch) or "negative" (decreasing stretch)
        """
        self.reads = readsobj.reads
        self.deriv = readsobj.blurred_deriv
        self.kind = kind

    def stretch_search(self):
        """
        Find all contiguous stretches of positive or negative derivative.

        Finds increasing stretches (blurred_deriv > 0) with `kind` = "positive".
        Finds decreasing stretches (blurred_deriv < 0) with `kind` = "negative".
        Note: considers values that are exactly 0 (rare) as both
            negative and positive.
        After finding all stretches, updates search_idx, search_starts,
            and search_sizes.
        """
        # Sizes of gaps between contiguous positive stretches = length of
        #     negative stretches
        kind = self.kind
        if kind == "positive":
            idx = np.where(self.deriv <= 0)[0]
        elif kind == "negative":
            idx = np.where(self.deriv >= 0)[0]
        else:
            raise ValueError("kind must be 'positive' or 'negative'")

        # Find discontinuity in indices
        idx_shift = idx[1:]  # indices shifted by one
        aligned = np.stack([idx[:-1], idx_shift])  # stack the two arrays
        stretch_starts = np.where(aligned[0] + 1 != aligned[1])[
            0
        ]  # list of indices (indices of aligned) where a gap begins
        stretch_sizes = aligned[1, stretch_starts] - aligned[0, stretch_starts]

        self.search_idx = idx
        self.search_starts = stretch_starts
        self.search_sizes = stretch_sizes

    def filter_stretch_length(self, min_gap):
        """
        Keep stretches that are longer than the `min_gap` argument.

        Determines the genomic locations of the sufficiently long stretches.
        Updates starts, sizes, start_datidx, start_genomidx, end_genomidx.
        """
        # filter features by length
        passing = np.where(self.search_sizes > min_gap)
        start_passing = self.search_starts[passing]
        sizes_passing = self.search_sizes[passing]

        # store indices of features
        # `datidx`: array of indices of the data (0 to n);
        #      n = length of the reads, cumulative sum, self.deriv, arrays
        # `genomidx`: array of actual genomic positions.
        start_datidx = self.search_idx[start_passing]  # indices of
        start_genomidx = self.reads[start_datidx, 0]  # genomic positions

        self.starts = start_passing
        self.sizes = sizes_passing
        self.start_datidx = start_datidx
        self.start_genomidx = start_genomidx
        self.end_genomidx = self.start_genomidx + self.sizes

    def ttest_stretches(self, gap_len=50, window_len=100, pval_cutoff=1):
        """
        Apply a t-test to the read density before and after each stretch.

        By default, this function is not used and the default p-value cutoff is set to 1
        to avoid filtering out any stretches on the basis of this test.
        """

        def ttest_candidate_shift(
            dat, start_datidx, stretch_len, gap_len, window_len, **kwargs
        ):
            """
            T-test an individual stretch of increasing/decreasing data.

            For a given stretch, look left and right (at some gap) of the strech
            and t-test the left vs. right to see if the increase/decrease
            is persistent and significant.
            """
            # set indices defining the left window of values and
            #     right window of values to test against each other
            left_start = start_datidx - gap_len - window_len
            left_end = start_datidx - gap_len
            right_start = start_datidx + stretch_len + gap_len
            right_end = start_datidx + stretch_len + gap_len + window_len

            # account for stretches very beginning and end of data
            if left_start < 0:
                left_start = 0
            if left_end < 0:
                left_end = 0
            if right_start > len(dat) - 1:
                right_start = len(dat) - 1
            if right_end > len(dat) - 1:
                right_end = len(dat) - 1

            left = dat[left_start:left_end, 1]
            right = dat[right_start:right_end, 1]

            t, p = stats.ttest_ind(
                left, right, equal_var=False, nan_policy="omit", **kwargs
            )

            return p

        # alternative hypothesis type
        if self.kind == "positive":
            alt = "less"
        elif self.kind == "negative":
            alt = "greater"
        else:
            raise ValueError("kind must be 'positive' or 'negative'")

        # apply ttest to all putative stretches
        results = []
        for idx, length in zip(self.start_datidx, self.sizes):
            p = ttest_candidate_shift(
                self.reads, idx, length, gap_len=50, window_len=100, alternative=alt
            )
            results.append(p)
        pvals = np.array(results)

        # filter based on p-value
        condition = pvals < pval_cutoff
        start = self.start_genomidx[condition]
        end = self.start_genomidx[condition] + self.sizes[condition]

        self.pvals = pvals
        self.start_genomidx = start
        self.end_genomidx = end

    def output_csv(self, filename, sort_length=True):
        """
        Create a Pandas dataframe of sufficiently long stretches.

        Output includes genomic position and stretch length. Sorts by stretch length
        (from long to short) by default. Outputs the dataframe as a .csv file.
        """
        # create the output dataframe
        compiled = np.stack([self.start_genomidx, self.sizes])
        compiled_df = pd.DataFrame(
            compiled.transpose(), columns=["genomic location", "length"]
        )

        # sort from longest to shortest features if desired
        if sort_length:
            compiled_df = compiled_df.sort_values("length", ascending=False)

        # output to csv
        compiled_df.to_csv(filename, index=False)


def arr_inrange(idx, feature_starts, feature_ends):
    """
    Locate indices within idx that lie between feature_starts and feature_ends.

    Parameters
    ----------
        - idx: an array of length p of position values
        - feature_starts: an array of length l of feature starting positions
        - feature_ends: an array of length l of feature ending positions

    Returns
    -------
        - inrange: a boolean array of length p
    """

    def isin_range(idx, feature_starts, feature_ends):
        """
        Check if a value lies between two corresponding bookends.

        Helper function for arr_inrange that gets vectorized.
        In the non-vectorized form, checks if a value (idx) lies between any
            two corresponding values of `feature_starts` and `feature_ends`
        """
        greater = np.where(feature_starts > idx)[
            0
        ]  # all instances of being greater than query
        if len(greater) == 0:
            return False
        first_greater = greater[0]
        start = feature_starts[first_greater - 1]
        end = feature_ends[first_greater - 1]
        if (idx > start) & (idx < end):
            return True
        else:
            return False

    # vectorize function and return vectorized result
    v_isin_range = np.vectorize(isin_range, excluded=["feature_starts", "feature_ends"])
    inrange = v_isin_range(
        idx=idx, feature_starts=feature_starts, feature_ends=feature_ends
    )
    return inrange


def output_to_wig(inc_stretchobj, dec_stretchobj, wig_output_filename, chrom):
    """
    Write a .wig file containing putative ramps and steps.

    Increasing stretches labeled with +1 and decreasing stretches are labeled with -1.

    Parameters
    ----------
        - inc_stretchobj: a StretchData object with increasing stretches
            (kind = 'positive')
        - dec_sretchobj: a StretchData object with decreasing stretches
            (kind = 'negative')
        - wig_output_filename: name of the output file
        - chrom: the appropriate chrom for the output .wig
    """
    # find which genomic indices lie between the starting and ending
    #     genomic indices for each feature
    dec_inrange = arr_inrange(
        idx=dec_stretchobj.reads[:, 0],
        feature_starts=dec_stretchobj.start_genomidx,
        feature_ends=dec_stretchobj.end_genomidx,
    )
    inc_inrange = arr_inrange(
        idx=inc_stretchobj.reads[:, 0],
        feature_starts=inc_stretchobj.start_genomidx,
        feature_ends=inc_stretchobj.end_genomidx,
    )

    # write wig: with increasing stretches as +1 and decreasing stretches as -1
    output_wig = dec_stretchobj.reads.copy()
    output_wig[:, 1] = 0
    output_wig[dec_inrange, 1] = -1
    output_wig[inc_inrange, 1] = 1
    print("Writing the ramps/steps file to a .wig")
    ff.write_wig(output_wig, wig_output_filename, chrom)
    print("Done writing the wig.")
