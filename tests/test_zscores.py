# -*- coding: utf-8 -*-
import sys

import pytest
from mock import patch
from numpy import append, array, mean, std
from numpy.testing import assert_array_almost_equal, assert_array_equal

from rendseq.file_funcs import write_wig
from rendseq.zscores import (
    _adjust_down,
    _adjust_up,
    _calc_score,
    _l_score_helper,
    _r_score_helper,
    _remove_outliers,
    main_zscores,
    parse_args_zscores,
    score_helper,
    validate_gap_window,
    z_score,
    z_scores,
)


@pytest.fixture
def reads():
    """Reads to use in multiple test cases"""
    return array(
        [
            [1, 5],
            [2, 6],
            [3, 8],
            [4, 10],
            [5, 8],
            [6, 10],
            [7, 1200],
            [8, 14],
            [9, 1],
            [10, 2],
            [11, 5],
            [12, 6],
            [109, 2],
            [208, 4],
        ]
    )


class TestMainAndParseArgsZscore:
    @pytest.fixture
    def regular_argslist(self):
        """A normal list of sys.argv[1:]"""
        return [
            "test_file",
            "--gap",
            "1",
            "--w_sz",
            "3",
            "--min_r",
            "0",
            "--save_file",
            False,
        ]

    def test_main(self, tmpdir, capfd, reads, regular_argslist):
        """Main with normal settings"""
        # Create a wig file with the reads() fixture
        file = tmpdir.join("file.txt")
        chrom = "test_chrom"
        write_wig(reads, file.strpath, chrom)

        # Modify the argslist with the temporary wig file
        regular_argslist = [""] + regular_argslist
        regular_argslist[1] = file.strpath

        # Run main with regular arguments
        with patch.object(sys, "argv", regular_argslist):
            main_zscores()
            out, err = capfd.readouterr()

        # Expect output
        assert out == "\n".join(
            [
                f"Calculating zscores for file {file.strpath}.",
                f"Ran zscores.py with the following settings:\ngap: 1, w_sz: 3,\nmin_r: 0, file_name: {file.strpath}\n",
            ]
        )

    def test_main_defaults(self, tmpdir, capfd):
        """Main with defaults"""
        # Create a wig file with the reads() fixture
        file = tmpdir.join("file.txt")
        chrom = "test_chrom"

        # Need a larger reads file for defaults
        reads = array([[a, b] for a, b in zip(range(1, 1000), range(1001, 2000))])
        write_wig(reads, file.strpath, chrom)

        # Modify the argslist with the temporary wig file
        argslist = ["", file.strpath]

        # Run main
        with patch.object(sys, "argv", argslist):
            main_zscores()
            out, err = capfd.readouterr()

        # Expect output
        assert out == "\n".join(
            [
                f"Calculating zscores for file {file.strpath}.",
                f'Wrote z_scores to {file.strpath[0:-8] + "Z_scores/file_zscores.wig"}',
                f"Ran zscores.py with the following settings:\ngap: 5, w_sz: 50,\nmin_r: 20, file_name: {file.strpath}\n",
            ]
        )

    def test_parse_args(self, regular_argslist):
        """Regular arguments"""
        args = parse_args_zscores(regular_argslist)
        assert args.filename == "test_file"
        assert args.gap == "1"
        assert args.w_sz == "3"
        assert args.min_r == "0"
        assert args.save_file == False

    def test_parse_args_defaults(self):
        """Makes sure the arg defaults are as-expected"""
        arg_list = ["test_file"]
        args = parse_args_zscores(arg_list)

        assert args.filename == "test_file"
        assert args.gap == 5
        assert args.w_sz == 50
        assert args.min_r == 20
        assert args.save_file


class TestZScores:
    def test_z_scores_regular(self, reads):
        """Z-scores of the reads fixture"""
        assert_array_almost_equal(
            z_scores(reads, gap=1, w_sz=3, min_r=0),
            array(
                [
                    [5, 0],
                    [6, -0.70262826],
                    [7, 202.20038777234487],
                    [8, 4.949747468305832],
                    [9, -0.7213550215235531],
                    [10, 0],
                ]
            ),
        )

    def test_z_scores_outlier(self, reads):
        """An outlier (near the edge where peaks aren't found) doesn't affect score"""
        reads[11] = [12, 1e8]
        assert_array_almost_equal(
            z_scores(reads, gap=1, w_sz=3, min_r=0),
            array(
                [
                    [5, 0],
                    [6, -0.70262826],
                    [7, 202.20038777234487],
                    [8, 4.949747468305832],
                    [9, -0.7213550215235531],
                    [10, 0],
                ]
            ),
        )

    def test_z_scores_highMin(self, reads):
        """Minimum r is too high for all reads"""
        assert_array_almost_equal(
            z_scores(reads, gap=1, w_sz=3, min_r=1e8),
            array([[5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0]]),
        )

    def test_z_scores_highMin_peak(self, reads):
        """Minimum r too high for all reads but the peak"""
        reads[6] = [7, 1e6]
        assert_array_almost_equal(
            z_scores(reads, gap=1, w_sz=3, min_r=1e6),
            array(
                [
                    [5, 0],
                    [6, -7.07101478e-01],
                    [7, 1.69298835e05],
                    [8, 0],
                    [9, -7.07123752e-01],
                    [10, -7.07127995e-01],
                ]
            ),
            decimal=4,
        )


class TestLWScoreHelper:
    def test_l_score_helper_nogap(self, reads):
        """No gap"""
        min_r = 0
        i = 2
        with pytest.warns(UserWarning):
            assert _l_score_helper(0, 1, min_r, reads, i) == score_helper(
                1, 2, min_r, reads, i
            )

    def test_l_score_helper_gap(self, reads):
        """A small gap"""
        min_r = 0
        i = 2
        assert _l_score_helper(1, 2, min_r, reads, i) == score_helper(
            0, 1, min_r, reads, i
        )

    def test_l_score_helper_largewindow(self, reads):
        """Window size is larger than array"""
        min_r = 0
        i = 2
        assert _l_score_helper(1, 100, min_r, reads, i) == score_helper(
            0, 1, min_r, reads, i
        )

    def test_r_score_helper_nogap(self, reads):
        """No gap"""
        min_r = 0
        i = 2
        with pytest.warns(UserWarning):
            assert _r_score_helper(0, 1, min_r, reads, i) == score_helper(
                2, 3, min_r, reads, i
            )

    def test_r_score_helper_gap(self, reads):
        """A small gap"""
        min_r = 0
        i = 2
        assert _r_score_helper(1, 109, min_r, reads, i) == score_helper(
            3, 12, min_r, reads, i
        )

    def test_r_score_helper_largewindow(self, reads):
        min_r = 0
        i = 2
        assert _r_score_helper(1, 1000, min_r, reads, i) == score_helper(
            3, 13, min_r, reads, i
        )


class TestValidateGapWindow:
    def test_window_zero(self):
        """Windows can't be zero"""
        with pytest.raises(ValueError) as e_info:
            validate_gap_window(100, 0)

        assert (
            e_info.value.args[0]
            == "Window size must be larger than 1 to find a z-score"
        )

    def test_window_gap_positive(self):
        """Windows and gaps can be positive"""
        try:
            validate_gap_window(100, 1)
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")

    def test_gap_negative(self):
        """Gaps can't be negative"""
        with pytest.raises(ValueError) as e_info:
            validate_gap_window(-1, 100)

        assert (
            e_info.value.args[0] == "Gap size must be at least zero to find a z-score"
        )

    def test_gap_zero(self):
        """Gaps can be zero, but should warn"""
        with pytest.warns(UserWarning):
            validate_gap_window(0, 100)


class TestScoreHelper:
    def test_score_helper_normal(self, reads):
        """score_helper is just a wrapper for calc_score if no outliers"""
        min_r = 1
        i = 1
        assert score_helper(0, 4, 0, reads, 1) == pytest.approx(
            _calc_score(list(reads[0:4, 1]), min_r, reads[i, 1])
        )

    def test_score_helper_outlier(self, reads):
        """score_helper with a clear outlier"""
        min_r = 1
        i = 1
        assert score_helper(0, 4, 0, reads + [800], 1) == pytest.approx(
            _calc_score(list(reads[0:4, 1]), min_r, reads[i, 1])
        )


class TestCalcScore:
    def test_calc_score_normal(self, reads):
        """Run-of-the-mill zscore"""
        assert _calc_score(reads[:, 1:], 2, reads[1, 1:]) == pytest.approx(-0.2780832)

    def test_calc_score_highmin(self, reads):
        """Reads don't hit minimimum background"""
        assert _calc_score(reads[:, 1:], 10000, 1) == None

    def test_calc_score_constant(self):
        """Reads are constant"""
        assert _calc_score(array([1] * 6), 0, 1) == 0

    def test_calc_score_empty(self):
        """Array is empty"""
        assert _calc_score(array([]), 0, 1) == None


class TestZScore:
    def test_z_score_normal(self, reads):
        """Run-of-the-mill zscore"""
        vals = reads[:, 1:]
        assert z_score(vals[1][0], mean(vals), std(vals)) == pytest.approx(-0.2780832)

    def test_z_score_constant(self):
        """Zero stdev"""
        assert z_score(1, 1, 0) == 0


class TestRemoveOutliers:
    def test_remove_outliers_empty(self):
        """Remove outliers from empty array"""
        assert_array_equal(_remove_outliers(array([])), array([]))

    def test_remove_outliers_homogen(self):
        """No outliers in homogenous array"""
        test_array = array([1] * 6)
        assert_array_equal(_remove_outliers(test_array), test_array)

    def test_remove_outliers_clearOutlier(self):
        """A clear outlier"""
        test_array = array([i for i in range(20)] + [80])
        assert_array_equal(_remove_outliers(test_array), test_array[0:-1])

    def test_remove_outliers_borderline(self):
        """Two values, near 2.5 stds away, but only one above"""
        test_array = append(array([1] * 10), [4.2, 5])

        assert_array_equal(_remove_outliers(test_array), test_array[0:-1])


class TestAdjustDown:
    def test_adjust_down_empty(self):
        """Can't adjust down empty reads"""
        with pytest.raises(ValueError) as e_info:
            _adjust_down(3, 0, array([]))

        assert e_info.value.args[0] == "requires non-empty reads"

    def test_adjust_down_inf(self, reads):
        """Target higher than all indicies"""
        assert _adjust_down(3, 1000, reads) == 3

    def test_adjust_down_onestep(self, reads):
        """Target one step away from current"""
        assert _adjust_down(2, 2, reads) == 1

    def test_adjust_down_multistep(self, reads):
        """Target multiple steps away from current"""
        assert _adjust_down(3, 2, reads) == 1

    def test_adjust_down_zero(self, reads):
        """Target lower than any index"""
        assert _adjust_down(1, -1, reads) == 0

    def test_adjust_down_oob(self, reads):
        """Current is higher than any index"""
        assert _adjust_down(5, 2, reads) == 1


class TestAdjustUp:
    def test_adjust_up_empty(self):
        """Can't adjust up empty reads"""
        with pytest.raises(ValueError) as e_info:
            _adjust_up(3, 0, array([]))

        assert e_info.value.args[0] == "requires non-empty reads"

    def test_adjust_up_onestep(self, reads):
        """Target one step away from current"""
        assert _adjust_up(1, 3, reads) == 2

    def test_adjust_up_multistep(self, reads):
        """Target multiple steps away from current"""
        assert _adjust_up(0, 3, reads) == 2

    def test_adjust_up_max(self, reads):
        """Target higher than any index"""
        assert _adjust_up(1, 1000, reads) == 13

    def test_adjust_up_oob(self, reads):
        """Current is lower than any index"""
        assert _adjust_up(2, 0, reads) == 2
