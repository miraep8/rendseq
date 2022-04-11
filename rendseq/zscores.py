'''
The z_scores.py module contains the code for transforming raw rendSeq data into
    z_score transformed data.  It also has many helper functions that assist in
    this calculation.
'''
import argparse
import warnings
from numpy import zeros, mean, std
from rendseq.file_funcs import write_wig, open_wig, make_new_dir, validate_reads

def adjust_down(cur_ind, target_val, reads):
    '''
    adjust_down is a helper function - will return the index of the lower read
        that is within range for the z-score calculation
    '''
    validate_reads(reads)

    cur_ind = min(cur_ind, len(reads)-1)
    while reads[cur_ind,0] > target_val:
        cur_ind -= 1

        if cur_ind == 0:
            break
    return cur_ind

def adjust_up(cur_ind, target_val, reads):
    '''
    adjust_up is a helper function - will return the index of the upper read
        that is within range for the z-score calculation
    '''
    if len(reads) < 1:
        raise ValueError("requires non-empty reads")

    cur_ind = min(max(cur_ind, 0), len(reads))

    while reads[cur_ind,0] < target_val:
        if cur_ind >= len(reads)-1:
            break

        cur_ind += 1

    return cur_ind

def z_score(val, v_mean, v_std):
    ''' 
    Calculates a z-score given a value, mean, and standard deviation 
        NOTE: Unlike a canonical z_score, the z_score() of a constant vector is 0
    '''
    score = 0 if v_std == 0 else (val - v_mean) / v_std
    return score

def remove_outliers(vals):
    '''
    remove_outliers will take a list of values and "normalize" it by trimming
        potentially extreme values.  We choose to remove rather than winsorize
        to avoid artificial deflation of the standard deviation. Extreme values
        are removed if they are greater than 2.5 std above the mean
    Parameters:
        -vals: an array of raw read values to be processed
    Returns:
        -new_v: another array of raw values which has had the extreme values
            removed.
    '''
    normalized_vals = vals
    if len(vals) > 1:
        v_mean = mean(vals)
        v_std = std(vals)
        if v_std != 0:
            normalized_vals = [v for v in vals if abs(z_score(v, v_mean, v_std)) < 2.5]

    return normalized_vals

def calc_score(vals, min_r, cur_val):
    '''
    calc_score will compute the z score (and first check if the std is zero).
    Parameters:
        -vals raw read count values array
        -min_r: the minumum number of reads needed to calculate score
        -cur_val: the value for which the z score is being calculated
    Returns:
        -score: the zscore for the current value, or None if insufficent reads
    '''
    score = None
    if sum(vals) > min_r:
        v_mean = mean(vals)
        v_std = std(vals)

        score = z_score(cur_val, v_mean, v_std)

    return score

def score_helper(start, stop, min_r, reads, i):
    ''' 
    Finds the z-score of reads[i] relative to the subsection of reads
        from start to stop, with a read cutoff of min_r
    '''
    reads_outlierless = remove_outliers(list(reads[start:stop, 1]))
    return calc_score(reads_outlierless, min_r, reads[i, 1])

def validate_window_gap(gap, w_sz):
    ''' Checks that gap and window size are reasonable in r/l_score_helper'''
    if w_sz < 1:
        raise ValueError("Window size must be larger than 1 to find a z-score")
    if gap < 0:
        raise ValueError("Gap size must be at least zero to find a z-score")
    if gap == 1:
        warnings.warn("Warning...a gap size of 1 includes the current position and may misrepresent peaks.")

def l_score_helper(gap, w_sz, min_r, reads, i):
    '''
    l_score_helper will find the indexes of reads to use for a z_score
        calculation with reads to the left of the current read, and will return
        the calculated score.
    '''
    validate_window_gap(gap, w_sz)
    l_start = adjust_up(i - (gap + w_sz), reads[i,0] - (gap + w_sz), reads)
    l_stop = adjust_up(i - gap, reads[i,0] - gap, reads)
    return score_helper(l_start, l_stop, min_r, reads, i)

def r_score_helper(gap, w_sz, min_r, reads, i):
    '''
    r_score_helper will find the indexes of reads to use for a z_score
        calculation with reads to the right of the current read, and will return
        the calculated score.
    '''
    validate_window_gap(gap, w_sz)
    r_start = adjust_down(i + gap, reads[i,0] + gap, reads)
    r_stop = adjust_down(i + gap + w_sz, reads[i,0] + gap + w_sz, reads)
    return score_helper(r_start, r_stop, min_r, reads, i)

def z_scores(reads, gap = 5, w_sz = 50, min_r = 20):
    '''
    z_scores will generate a companion z_score file based on the local data
        around each read in the read file.
    Parameters:
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
    Returns:
        -z_score (2xn array): a 2xn array with the first column being position
            and the second column being the z_score.
    '''

    #make array of zscores - same length as raw reads:
    z_score = zeros([len(reads) - 2*(gap + w_sz),2])
    z_score[:,0] = reads[gap + w_sz:len(reads) - (gap + w_sz),0]
    for i in range((gap + w_sz + 1),(len(reads) - (gap + w_sz))):
        # calculate the z score with values from the left:
        l_score = l_score_helper(gap, w_sz, min_r, reads, i)
        # calculate z score with reads from the right:
        r_score = r_score_helper(gap, w_sz, min_r, reads, i)
        # set the zscore to be the smaller valid score of the left/right scores:
        if l_score is None and r_score is None: #if there were insufficient reads on both sides
            z_score[i-(gap + w_sz),1] = reads[i,1]/1.5
        elif (not r_score is None) and (l_score is None or abs(r_score) < abs(l_score)):
            z_score[i-(gap + w_sz),1] = r_score
        elif (not l_score is None) and (r_score is None or abs(l_score) < abs(r_score)):
            z_score[i-(gap + w_sz),1] = l_score
    return z_score

def main():
    ''' 
    Process command line arguments and run Z-score calculations.
    Effect: Writes messages to standard out. If --save-file flag,
    also writes output to disk.
    '''
    parser = argparse.ArgumentParser(description = 'Takes raw read file and\
                                        makes a modified z-score for each\
                                        position. Takes several optional\
                                        arguments')
    parser.add_argument("filename", help = "Location of the raw_reads file that\
                                        will be processed using this function.\
                                        Should be a properly formatted wig\
                                        file.")
    parser.add_argument("--gap", help = "gap (interger):   number of reads\
                                        surround the current read of interest\
                                        that should be excluded in the z_score\
                                        calculation. Defaults to 5.",
                                default = 5)
    parser.add_argument("--w_sz", help = "w_sz (integer): the max dis (in nt)\
                                        away from the current position one\
                                        should include in zscore calulcation.\
                                        Default to 50.",
                                default = 50)
    parser.add_argument("--min_r", help = "min_r (integer): density threshold.\
                                        If there are less than this number of\
                                        reads going into the z_score\
                                        calculation for a point that point is\
                                        excluded.  note this is sum of reads in\
                                        the window.  Default is 20",
                                default = 20)
    parser.add_argument("--save_file", help = "Save the z_scores file as a new\
                                        wig file in addition to returning the\
                                        z_scores.  Default = True",
                                default = True)
    args = parser.parse_args()
    filename = args.filename
    print(f'Calculating zscores for file {filename}.')
    reads, chrom = open_wig(filename)
    z_score = z_scores(reads, gap = args.gap, w_sz = args.w_sz,
                min_r = args.min_r)
    if args.save_file:
        file_loc = filename[:filename.rfind('/')]
        z_score_dir = make_new_dir([file_loc, '/Z_scores/'])
        file_start = filename[filename.rfind('/'):filename.rfind('.wig')]
        z_score_file = ''.join([z_score_dir, file_start, '_zscores.wig'])
        write_wig(z_score, z_score_file, chrom)
    print(f'Ran zscores.py with the following settings: \
        gap: {args.gap}, w_sz: {args.w_sz}, min_r: {args.min_r},\
        file_name: {args.filename} ')


if __name__ == '__main__':
    main()
