'''
peaks.py will take a zscore and find the peaks in it using the vertibi algorithm.
'''
import argparse
from math import log, inf
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from file_funcs import open_wig, write_wig, make_new_dir

def populate_trans_mat(z_scores, peak_center, spread, trans_m, states):
    '''
    populate_trans_mat will calculate the values for the transition matrix that
        is used in the vertibi algortihm to find the optimal path.
    Paramters:
        -z_scores (2xn array): - required: first column is position (ie bp
            location) second column is a modified z_score for that position.
        -peak_center (float): the mean of the emission probability distribution
            for the peak state.
        -spread (float): the standard deviation of the peak emmission
            distribution.
        -trans_m (matrix): the transition probabilities between states.
        -states (matrix): how internal and peak are represented in the wig file
    '''
    print('Calculating Transition Matrix')
    trans_1 = np.zeros([len(states), len(z_scores)])
    trans_2 = np.zeros([len(states), len(z_scores)]).astype(int)
    trans_1[:,0] = 1
    #Vertibi Algorithm:
    for i in range(1,len(z_scores)):
        #emission probabilities:
        probs = [norm.pdf(z_scores[i,1]), norm.pdf(z_scores[i,1], peak_center, spread)]
        # we use log probabilities for computational reasons. -Inf means 0 probability
        for j in range(len(states)):
            paths = np.zeros([len(states), 1])
            for k in range(len(states)):
                if (trans_1[k, i-1] == -inf) or (trans_m[k,j] == 0) or (probs[j] == 0):
                    paths[k] = -inf
                else:
                    paths[k] = trans_1[k, i-1] + log(trans_m[k, j]) + log(probs[j])
            trans_2[j,i] = np.argmax(paths)
            trans_1[j,i] = paths[trans_2[j,i]]
    return trans_1, trans_2

def hmm_peaks(z_scores, i_to_p = 1/1000, p_to_p = 1/1.5, peak_center = 10, spread = 2):
    '''
    hmm_peaks implements the vertibi algorithm to fit a HMM to the data given
        the z_scores of the raw data.
    Parameters:
        -z_scores (2xn array): - required: first column is position (ie bp
            location) second column is a modified z_score for that position.
        -i_to_p (float): value should be between zero and 1, represents
            probability of transitioning from inernal state to peak state. The
            default value is 1/2000, based on asseumption of geometrically
            distributed transcript lengths with mean length 2000. Should be a
            robust parameter.
        -p_to_p (float): The probability of a peak to peak transition.  Default
            1/1.5.
        -peak_center (float): the mean of the emission probability distribution
            for the peak state.
        -spread (float): the standard deviation of the peak emmission
            distribution.
    Returns:
        -peaks: a 2xn array with the first column being position and the second
            column being a peak assignment.
   '''
    print('Finding Peaks')
    trans_m = np.asarray([[(1 - i_to_p),(i_to_p)],[p_to_p,(1 - p_to_p)]]) #transition probability
    peaks = np.zeros([len(z_scores),2])
    peaks[:,0] = z_scores[:,0]
    states = [1,100] # how internal and peak are represented in the wig file
    trans_1, trans_2 = populate_trans_mat(z_scores, peak_center, spread, trans_m, states)
    #Now we trace backwards and find the most likely path:
    max_inds = np.zeros([len(peaks)]).astype(int)
    max_inds[len(peaks) - 1] = int(np.argmax(trans_1[:, len(trans_1)]))
    peaks[1,-1] = states[max_inds[len(peaks)-1]]
    for index in reversed(list(range(len(peaks)))):
        max_inds[index - 1] = trans_2[max_inds[index], index]
        peaks[index - 1, 1] = states[max_inds[index - 1]]
    print(f'Found {sum(peaks[:,1] > 1)} Peaks')
    return peaks

def make_kink_fig(save_file, seen, exp, pnts):
    '''
    make_kink_fig is a helper function which creates a figure comparing the
        number of observed positions with a z score equal to or greater than a
        threshold versus the expected number of positions with that z score.
    Parameters:
        - save_fig - the name of the file to save the plot to.
        - seen - the obs number of positions with a given z score or greater
        - exp - the exp number of positions with a given z score or greater
        - pnts - the z score values/x axis of the plot.
    '''
    plt.plot(pnts, seen, label = 'Observed')
    plt.plot(pnts, exp, label = 'Expected')
    plt.yscale('log')
    plt.ylabel('Number of Positions with Z score Greater than or equal to')
    plt.xlabel('Z score')
    plt.legend()
    plt.savefig(save_file)

def calc_thresh(z_scores, method):
    '''
    calc_thresh will calculate and appropriate threshold for a threshold based
        peak calling method if one was not supplied. It can automatically
        select this threshold based on several methods (which the user can
        choose between).
    Parameters:
        - z_scores (2xn array): the calculated z scores, where the first column
            represents the nt position and the second represents a z score.
        - method (string): the name of the threshold calculating method to use.
    Returns:
        - threshold (float): the calculated threshold.
    '''
    methods = ['expected_val', 'kink']
    thresh = 15
    if method == 'expected_val': #threshold such that num peaks exp < 1.
        p_val = 1/len(z_scores) #note this method is dependent on genome size
        thresh = round(norm.ppf(1-p_val), 1)
    elif method == 'kink':  #where the num z_scores exceeds exp num by 1000x
        num_exceed = 1000
        pnts = np.arange(0, 20, .1)
        seen = [0 for i in range(len(pnts))]
        exp = [0 for i in range(len(pnts))]
        thresh = -1
        for ind, point in enumerate(pnts):
            seen[ind] = np.sum(z_scores[:,1] > point)
            exp[ind] = (1 - norm.cdf(point))*len(z_scores)
            if seen[ind] >= num_exceed*exp[ind] and thresh == -1:
                thresh = point
        make_kink_fig('./kink.png', seen, exp, pnts)

    else:
        print(f'The method selected ({method}) does not match one of the \
                supported methods.  Please select one from {methods}.  \
                Defaulting to threshold of {thresh}')
    return thresh

def thresh_peaks(z_scores, thresh = None, method = 'kink'):
    '''
    thresh_peaks will calculate a peaks from the provided z scores.  This
        approach will count all positions with a z score above a certain
        threshold as a peak. (and can also calculate this threshold)
    Parameters:
        - z_scores - a 2xn array of nt positions and zscores at that pos.
        - thresh - the threshold value to use.  If none is provided it will be
            automatically calculated.
        - method - the method to use to automatically calculate the z score
            if none is provided.  Default method is "kink"
    '''
    if thresh is None:
        thresh = calc_thresh(z_scores, method)
    peaks = np.zeros([len(z_scores),2])
    peaks[:,0] = z_scores[:,0]
    peaks[:,1] = (z_scores[:,1] > thresh).astype(int)
    return peaks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Can run from the\
                                        commmand line.  Please pass a \
                                        zscore file and select a method \
                                        for peak fitting.')
    parser.add_argument("filename", help = "Location of the zscore file")
    parser.add_argument("method", help = "User must pass the desired peak\
                                        fitting method.  Choose \"thesh\" \
                                        or \"hmm\"")
    parser.add_argument("--save_file", help = "Save the z_scores file as a new\
                                        wig file in addition to returning the\
                                        z_scores.  Default = True",
                                default = True)
    ## TODO: add more optional args to parser - like zscores.py
    args = parser.parse_args()
    filename = args.filename
    z_scores, chrom = open_wig(filename)
    if args.method == "thresh":
        print(f'Using the thresholding method to find peaks for {filename}')
        peaks = thresh_peaks(z_scores)
    elif args.method == "hmm":
        print(f'Using the hmm method to find peaks for {filename}')
        peaks = hmm_peaks(z_scores)
    else:
        print('Issue!  Must pass a valid peak finding method!')
    if args.save_file:
        file_loc = filename[:filename.rfind('/')]
        file_loc = file_loc[:file_loc.rfind('/')]
        peak_dir = make_new_dir([file_loc, '/Peaks/'])
        file_start = filename[filename.rfind('/'):filename.rfind('.wig')]
        peak_file = ''.join([peak_dir, file_start, '_peaks.wig'])
        write_wig(peaks, peak_file, chrom)
