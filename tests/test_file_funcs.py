import pytest
from mock import patch
from numpy import array
from numpy.testing import assert_array_equal
from rendseq.file_funcs import make_new_dir, open_wig, write_wig


class TestReadWrite:
    def test_write_wig(self, tmpdir):
        file = tmpdir.join("file.txt")
        chrom = "test_chrom"
        reads = array([[1, 7], [0, 0], [0, 1], [1, 0]])
        write_wig(reads, file.strpath, chrom)

        assert file.read() == "\n".join(
            ["track type=wiggle_0", "variableStep chrom=test_chrom", "1\t7", "1\t0\n"]
        )

    def test_open_wig(self, tmpdir):
        file = tmpdir.join("file.txt")
        with open(file, "w") as wigFH:
            wigFH.write("track type=wiggle_0\n")
            wigFH.write("variableStep chrom=test_chrom\n")
            wigFH.write("1\t5\n2\t6\n3\t8\n109\t1")

        reads, chrom = open_wig(file.strpath)

        assert chrom == "test_chrom"
        assert_array_equal(reads, array([[1, 5], [2, 6], [3, 8], [109, 1]]))

    def test_open_then_write(self, tmpdir):
        """If you read a file, and then write it, it should look the same"""
        read_file = tmpdir.join("read_file.txt")
        with open(read_file, "w") as wigFH:
            wigFH.write("track type=wiggle_0\n")
            wigFH.write("variableStep chrom=test_chrom\n")
            wigFH.write("1\t5\n2\t6\n3\t8\n109\t1\n")

        reads, chrom = open_wig(read_file.strpath)

        write_file = tmpdir.join("write_file.txt")
        write_wig(reads, write_file.strpath, chrom)

        assert read_file.read() == write_file.read()


@patch("rendseq.file_funcs.mkdir")
def test_make_new_dir(mock_mkdir):
    make_new_dir(["hello", "_rendseq", "_world"])
    mock_mkdir.assert_called_once_with("hello_rendseq_world")
