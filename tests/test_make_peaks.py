# -*- coding: utf-8 -*-
import sys
from os import remove
from os.path import exists

import pytest
from mock import patch
from numpy import array, ndim, where
from numpy.random import normal
from numpy.testing import assert_array_almost_equal, assert_array_equal

from rendseq.file_funcs import write_wig
from rendseq.make_peaks import (
    _calc_thresh,
    _make_kink_fig,
    _populate_trans_mat,
    hmm_peaks,
    main_make_peaks,
    parse_args_make_peaks,
    thresh_peaks,
)


@pytest.fixture
def z_scores():
    """Define some random z-scores"""
    return array(
        [[loc, z] for loc, z in zip(range(1, 1000), normal(0, 1, size=(1, 1000))[0])]
    )


# Helper function
def clean_kink():
    if exists("./kink.png"):
        remove("./kink.png")


class TestParseArgsAndMain:
    @pytest.fixture
    def regular_argslist(self):
        """A normal list of sys.argv[1:]"""
        return [
            "test_file",
            "thresh",
            "--save_file",
            False,
        ]

    def test_main_thresh(self, tmpdir, capfd, regular_argslist, z_scores):
        """Main with regular arguments"""

        # Create a wig file with the z_scores() fixture
        file = tmpdir.join("file.txt")
        chrom = "test_chrom"
        write_wig(z_scores, file.strpath, chrom)

        # Modify the argslist with the temporary wig file
        regular_argslist = [""] + regular_argslist
        regular_argslist[1] = file.strpath

        # Run the main function
        with patch.object(sys, "argv", regular_argslist):
            main_make_peaks()
            clean_kink()
            out, err = capfd.readouterr()

        assert (
            out == f"Using the thresholding method to find peaks for {file.strpath}\n"
        )

    def test_main_hmm(self, tmpdir, capfd, regular_argslist, z_scores):
        """Main with hmm"""

        # Create a wig file with the z_scores() fixture
        file = tmpdir.join("file.txt")
        chrom = "test_chrom"
        write_wig(z_scores, file.strpath, chrom)

        # Modify the argslist with the temporary wig file
        regular_argslist = [""] + regular_argslist
        regular_argslist[1] = file.strpath
        regular_argslist[2] = "hmm"

        # Run the main function
        with patch.object(sys, "argv", regular_argslist):
            main_make_peaks()
            clean_kink()
            out, err = capfd.readouterr()

        assert out == "\n".join(
            [
                f"Using the hmm method to find peaks for {file.strpath}",
                "Finding Peaks",
                "Calculating Transition Matrix",
                "Found 0 Peaks\n",
            ]
        )

    def test_main_undefined(self, tmpdir, capfd, regular_argslist, z_scores):
        """Main with undefined method"""

        # Create a wig file with the z_scores() fixture
        file = tmpdir.join("file.txt")
        chrom = "test_chrom"
        write_wig(z_scores, file.strpath, chrom)

        # Modify the argslist with the temporary wig file
        regular_argslist = [""] + regular_argslist
        regular_argslist[1] = file.strpath
        regular_argslist[2] = "undefined"

        # Run the main function
        with patch.object(sys, "argv", regular_argslist):
            with pytest.raises(ValueError) as e_info:
                main_make_peaks()
                clean_kink()
                assert (
                    e_info.value.args[0]
                    == "undefined is not a valid peak finding method, see --help"
                )

    def test_main_defaults(self, tmpdir, capfd, z_scores):
        # Create a wig file with the z_scores() fixture
        file = tmpdir.join("file.wig")
        chrom = "test_chrom"
        write_wig(z_scores, file.strpath, chrom)

        # Modify the argslist with the temporary wig file
        default_argslist = ["", file.strpath, "thresh"]

        # Run the main function
        with patch.object(sys, "argv", default_argslist):
            main_make_peaks()
            clean_kink()
            assert exists(file.strpath[0:-8].replace("\\", "/") + "Peaks/")
            outfile = file.strpath[0:-8].replace("\\", "/") + "Peaks/file_peaks.wig"
            assert exists(outfile)
            out, err = capfd.readouterr()
            assert out == "\n".join(
                [
                    f"Using the thresholding method to find peaks for {file.strpath}",
                    f"Wrote peaks to {outfile}\n",
                ]
            )

    def test_parse_args(self, regular_argslist):
        """Regular arguments"""
        args = parse_args_make_peaks(regular_argslist)
        assert args.filename == "test_file"
        assert args.method == "thresh"
        assert not args.save_file

    def test_parse_args_defaults(self):
        """Makes sure the arg defaults are as-expected"""
        arg_list = ["test_file", "thresh"]
        args = parse_args_make_peaks(arg_list)

        assert args.filename == "test_file"
        assert args.method == "thresh"
        assert args.save_file


class TestPopulateTransMat:
    # TODO: Need more detailed tests
    def test_populate_trans_mat(self, capfd, z_scores):
        matricies = _populate_trans_mat(
            z_scores, 10, 2, array([[0.5, 0.5], [0.5, 0.5]]), [1, 100]
        )

        assert ndim(matricies) == 3

        # Test print output
        out, err = capfd.readouterr()
        assert out == "Calculating Transition Matrix\n"


class TestHmmPeaks:
    def test_hmm_peaks(self, capfd, z_scores):
        """A regular set of z scores with a peak"""
        z_scores[500, 1] = 5
        peaks_almost_1 = array([[loc, z] for loc, z in zip(range(1, 1000), [1] * 1000)])
        peaks_almost_1[500, 1] = 100
        assert_array_equal(hmm_peaks(z_scores), peaks_almost_1)

        # Test print output
        out, err = capfd.readouterr()
        assert out == "Finding Peaks\nCalculating Transition Matrix\nFound 1 Peaks\n"

    def test_hmm_peaks_extremePeak_notinCenter(self, capfd, z_scores):
        """A regular set of z scores with an extreme peak"""
        z_scores[500, 1] = 10e4
        peaks_almost_1 = array([[loc, z] for loc, z in zip(range(1, 1000), [1] * 1000)])
        peaks_almost_1[500, 1] = 100

        assert_array_equal(hmm_peaks(z_scores), peaks_almost_1)

        # Test print output
        out, err = capfd.readouterr()
        assert out == "Finding Peaks\nCalculating Transition Matrix\nFound 1 Peaks\n"

    def test_hmm_peaks_bad_parameters(self, z_scores):
        """A regular set of z scores with a peak"""
        z_scores[500, 1] = 12
        with pytest.raises(ValueError):
            hmm_peaks(z_scores, i_to_p=0.5, p_to_p=0.99, peak_center=100, spread=0.5)

    def test_hmm_peaks_p_to_p_is_one(self, z_scores):
        """A regular set of z scores with a peak"""
        NUM_PNTS = 1000
        z_scores = array(
            [
                [loc, z]
                for loc, z in zip(
                    range(1, NUM_PNTS), normal(10, 0.1, size=(1, NUM_PNTS - 1))[0]
                )
            ]
        )
        all_peaks = array(
            [[loc, z] for loc, z in zip(range(1, NUM_PNTS), [100] * NUM_PNTS)]
        )

        assert_array_equal(hmm_peaks(z_scores, peak_center=10, spread=0.1), all_peaks)

    # TODO: Tests for all of the hmm_peaks parameters


def test_make_kink_fig(tmpdir):
    """Just make sure it makes a plot file"""
    file = tmpdir.join("file")
    _make_kink_fig(file.strpath + ".png", [0, 1], [1, 0], [1, 0], [1, 2])
    assert exists(file.strpath + ".png")


class TestThreshPeaks:
    def test_thresh_peaks_highThresh(self, z_scores):
        """Very high threshold"""
        z_scores_0 = array([[loc, z] for loc, z in zip(range(1, 1000), [0] * 1000)])
        assert_array_equal(thresh_peaks(z_scores, thresh=1e9), z_scores_0)

    def test_thresh_peaks_lowThreshold(self, z_scores):
        """A very low threshold"""
        z_scores_1 = array([[loc, z] for loc, z in zip(range(1, 1000), [1] * 1000)])
        assert_array_almost_equal(thresh_peaks(z_scores, thresh=-1e9), z_scores_1)

    def test_thresh_peaks_threshold(self, z_scores):
        """Test the filtering: threshold is exactly at one point"""
        z_scores_almost0 = array(
            [[loc, z] for loc, z in zip(range(1, 1000), [0] * 1000)]
        )
        max_score = max(z_scores[:, 1])
        thresh = max_score - 1e-6

        z_scores_almost0[where(z_scores == max_score)[0], 1] = 1

        assert_array_almost_equal(
            thresh_peaks(z_scores, thresh=thresh), z_scores_almost0
        )


class TestCalcThresh:
    def test_calc_thresh_default(self, z_scores):
        """Threshold with invalid thresh procedure"""
        with pytest.warns(UserWarning):
            assert _calc_thresh(z_scores, "") == 15

    def test_calc_thresh_expected_val(self, z_scores):
        """Expected_val threshold"""
        assert _calc_thresh(z_scores, "expected_val") == pytest.approx(3.1)

    def test_calc_thresh_kink(self, tmpdir, z_scores):
        """Kink threshold"""
        file = tmpdir.join("file.txt")
        kink_file = file.strpath[0:-8] + "test_kink.png"
        assert _calc_thresh(z_scores, "kink", kink_file) == pytest.approx(8.3)
        assert exists(kink_file)
