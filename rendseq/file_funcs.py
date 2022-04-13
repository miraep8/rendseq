# -*- coding: utf-8 -*-
"""Functions for fetching, creating, and opening raw and processed data files."""

from os import mkdir, path

from numpy import asarray, delete, where
from pandas import read_csv


def write_wig(wig_track, wig_file_name, chrom_name):
    """Write provided data to the wig file.

    Parameters
    ----------
        - wig_track (required) - the wig data you wish to write (in 2xn array)
        - wig_file_name (string) - the new file you will write to
    """
    d_inds = where(wig_track[:, 0] < 1)
    delete(wig_track, d_inds)
    with open(wig_file_name, "w+", encoding="utf-8") as wig_file:
        wig_file.write("track type=wiggle_0\n")
        wig_file.write(f"variableStep chrom={chrom_name}\n")
        for i in range(len(wig_track)):
            wig_file.write(f"{int(wig_track[i,0])}\t{wig_track[i,1]}\n")


def open_wig(filename):
    """Open the provided wig file and return the contents into a 2xn array.

    Parameters
    ----------
        -filename (string) - required: the string containing the location of
            the filename you desire to open!

    Returns
    -------
        -reads (2xn array): a 2xn array with the first column being position
            and the second column being the count at that position (raw read,
            z_score etc)
    """
    # first we will read the chrom from the second line in the wig file:
    with open(filename, "r", encoding="utf8") as file:
        line = file.readline()
        line = file.readline()
        chrom = line[line.rfind("=") + 1 :]
    # next we read all the wig file data and return that:
    reads = asarray(read_csv(filename, sep="\t", header=1, names=["bp", "count"]))
    return reads, chrom


def make_new_dir(dir_parts):
    """Create a new directory and return valid path to it.

    Parameters
    ----------
        - dir_parts  - a list of strings to be joined to make the directory name

    Returns
    -------
        - dir_str - the directory name
    """
    dir_str = "".join(dir_parts)
    if not path.isdir(dir_str):
        mkdir(dir_str)
    return dir_str
