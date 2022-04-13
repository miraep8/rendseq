# -*- coding: utf-8 -*-
"""Functions for fetching, creating, and opening raw and processed data files."""

from os import mkdir, path

from numpy import asarray, delete, where
from pandas import read_csv


def validate_reads(reads):
    """
    Parameters:
        -reads (2xn array): a 2xn array with the first column being position
            and the second column being the count at that position (raw read,
            z_score etc)
    Returns:
        NoneType
    Effects:
        Throws exceptions if reads are not correctly formatted
    """
    # Check if reads are empty
    if len(reads) < 1:
        raise ValueError("requires non-empty reads")

    # If reads.shape doesn't work, then it's not an np array
    try:
        shape = reads.shape
    except Exception as e:
        raise ValueError(f"reads must be numpy array, not {type(reads)}")

    # Must have two columns
    if shape[1] != 2:
        raise ValueError(f"reads must be (n,2), not {reads.shape}")


def write_wig(wig_track, wig_file_name, chrom_name):
   """Write provided data to the wig file.

    Parameters
    ----------
        - wig_track (required) - the wig data you wish to write (in 2xn array)
        - wig_file_name (string) - the new file you will write to
    """
    validate_reads(wig_track)
    d_inds = where(wig_track[:, 0] < 1)
    wig_track = delete(wig_track, d_inds, axis=0)
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
        try:
            next(file)
        except StopIteration as e:
            raise ValueError(f"{filename} appears to have zero lines")

        line = file.readline()
        chrom = line[line.rfind("=") + 1 :].rstrip()
    # next we read all the wig file data and return that if it's valid:
    reads = asarray(read_csv(filename, sep="\t", header=1, names=["bp", "count"]))
    validate_reads(reads)
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
    if not isdir(dir_str):
        mkdir(dir_str)
    return dir_str
