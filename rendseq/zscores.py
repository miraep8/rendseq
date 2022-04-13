# -*- coding: utf-8 -*-
"""Functions needed for z-score transforming raw rendSeq data."""

import argparse

from numpy import mean, std, zeros

from rendseq.file_funcs import make_new_dir, open_wig, write_wig


def _adjust_down(cur_ind, target_val, reads):
    """Calculate the lower reads index in range for the z-score calculation."""
    cur_ind = min(cur_ind, len(reads) - 1)
    while reads[cur_ind, 0] > target_val:
        cur_ind -= 1
    return cur_ind


def _adjust_up(cur_ind, target_val, reads):
    """Calculate the higher reads index in range for the z-score calculation."""
    cur_ind = max(cur_ind, 0)
    while reads[cur_ind, 0] < target_val:
        cur_ind += 1
    return cur_ind


def _remove_outliers(vals):
    """Normalize window of reads by removing outliers (values 2.5 std > mean).

    Parameters
    ----------
        -vals: an array of raw read values to be processed

    Returns
    -------
        -new_v: another array of raw values which has had the extreme values
            removed.
    """
    if len(vals) > 1:
        new_v = []
        v_mean = mean(vals)
        v_std = std(vals)
        for value in vals:
            if v_std == 0 or abs((value - v_mean) / v_std) < 2.5:
                new_v.append(value)
    else:
        new_v = vals
    return new_v


def _calc_score(vals, min_r, cur_val):
    """Compute the z score.

    Parameters
    ----------
        -vals raw read count values array
        -min_r: the minumum number of reads needed to calculate score
        -cur_val: the value for which the z score is being calculated

    Returns
    -------
        -score: the zscore for the current value, or None if insufficent reads
    """
    score = None
    if sum(vals) > min_r:
        v_mean = mean(vals)
        v_std = std(vals)
        if not v_std == 0:
            score = (cur_val - v_mean) / v_std
        else:
            score = cur_val - v_mean
    return score


def _l_score_helper(gap, w_sz, min_r, reads, i):
    """Find the z_score based on reads to the left of the current pos."""
    l_start = _adjust_up(i - (gap + w_sz), reads[i, 0] - (gap + w_sz), reads)
    l_stop = _adjust_up(i - gap, reads[i, 0] - gap, reads)
    l_vals = _remove_outliers(list(reads[l_start:l_stop, 1]))
    l_score = _calc_score(l_vals, min_r, reads[i, 1])
    return l_score


def _r_score_helper(gap, w_sz, min_r, reads, i):
    """Find the z_score based on reads to the right of the current pos."""
    r_start = _adjust_down(i + gap, reads[i, 0] + gap, reads)
    r_stop = _adjust_down(i + gap + w_sz, reads[i, 0] + gap + w_sz, reads)
    r_vals = _remove_outliers(list(reads[r_start:r_stop, 1]))
    r_score = _calc_score(r_vals, min_r, reads[i, 1])
    return r_score


def z_scores(reads, gap=5, w_sz=50, min_r=20):
    """Perform modified z-score transformation of reads.

    Parameters
    ----------
        -reads 2xn array - raw rendseq reads
        -gap (interger):   number of reads surround the current read of
            interest that should be excluded in the z_score calculation.
        -w_sz (integer): the max distance (in nt) away from the current position
            one should include in zscore calulcation.
        -min_r (integer): density threshold. If there are less than this number
            of reads going into the z_score calculation for a point that point
            is excluded.  note this is sum of reads in the window
        -file_name (string): the base file_name, can be passed in to customize
            the message printed

    Returns
    -------
        -z_score (2xn array): a 2xn array with the first column being position
            and the second column being the z_score.
    """
    # make array of zscores - same length as raw reads:
    z_score = zeros([len(reads) - 2 * (gap + w_sz), 2])
    z_score[:, 0] = reads[gap + w_sz : len(reads) - (gap + w_sz), 0]
    for i in range((gap + w_sz + 1), (len(reads) - (gap + w_sz))):
        # calculate the z score with values from the left:
        l_score = _l_score_helper(gap, w_sz, min_r, reads, i)
        # calculate z score with reads from the right:
        r_score = _r_score_helper(gap, w_sz, min_r, reads, i)
        # set the zscore to be the smaller valid score of the left/right scores:
        if (
            l_score is None and r_score is None
        ):  # if there were insufficient reads on both sides
            z_score[i - (gap + w_sz), 1] = reads[i, 1] / 1.5
        elif r_score and (l_score is None or abs(r_score) < abs(l_score)):
            z_score[i - (gap + w_sz), 1] = r_score
        elif l_score and (r_score is None or abs(l_score) < abs(r_score)):
            z_score[i - (gap + w_sz), 1] = l_score
    return z_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Takes raw read file and\
                                        makes a modified z-score for each\
                                        position. Takes several optional\
                                        arguments"
    )
    parser.add_argument(
        "filename",
        help="Location of the raw_reads file that\
                                        will be processed using this function.\
                                        Should be a properly formatted wig\
                                        file.",
    )
    parser.add_argument(
        "--gap",
        help="gap (interger):   number of reads\
                                        surround the current read of interest\
                                        that should be excluded in the z_score\
                                        calculation. Defaults to 5.",
        default=5,
    )
    parser.add_argument(
        "--w_sz",
        help="w_sz (integer): the max dis (in nt)\
                                        away from the current position one\
                                        should include in zscore calulcation.\
                                        Default to 50.",
        default=50,
    )
    parser.add_argument(
        "--min_r",
        help="min_r (integer): density threshold.\
                                        If there are less than this number of\
                                        reads going into the z_score\
                                        calculation for a point that point is\
                                        excluded.  note this is sum of reads in\
                                        the window.  Default is 20",
        default=20,
    )
    parser.add_argument(
        "--save_file",
        help="Save the z_scores file as a new\
                                        wig file in addition to returning the\
                                        z_scores.  Default = True",
        default=True,
    )
    args = parser.parse_args()
    filename = args.filename
    print(f"Calculating zscores for file {filename}.")
    reads, chrom = open_wig(filename)
    z_score = z_scores(reads, gap=args.gap, w_sz=args.w_sz, min_r=args.min_r)
    if args.save_file:
        file_loc = filename[: filename.rfind("/")]
        z_score_dir = make_new_dir([file_loc, "/Z_scores/"])
        file_start = filename[filename.rfind("/") : filename.rfind(".wig")]
        z_score_file = "".join([z_score_dir, file_start, "_zscores.wig"])
        write_wig(z_score, z_score_file, chrom)
    print(
        f"Ran zscores.py with the following settings: \
        gap: {args.gap}, w_sz: {args.w_sz}, min_r: {args.min_r},\
        file_name: {args.filename} "
    )
