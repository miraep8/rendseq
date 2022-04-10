import pytest
from mock import patch
from numpy import array
from numpy.testing import assert_array_equal
from rendseq.file_funcs import make_new_dir, open_wig, write_wig, validate_reads


class TestValidateReads:
    def test_correct(self):
        """validate a correct read array"""
        try:
            validate_reads(array([[1, 2], [3, 4]]))
        except Exception as e:
            assert False, f"validate_reads invalid exception: {e}"

    def test_incorrect_dim(self):
        """read array has too many columns"""
        with pytest.raises(ValueError) as e_info:
            validate_reads(array([[1, 2, 3], [4, 5, 6]]))

        assert e_info.value.args[0] == "reads must be (n,2), not (2, 3)"

    def test_incorrect_type(self):
        """read array isn't actually an array"""
        with pytest.raises(ValueError) as e_info:
            validate_reads([1, 2, 3])

        assert e_info.value.args[0] == "reads must be numpy array, not <class 'list'>"


class TestOpenWriteWig:
    def test_write_wig(self, tmpdir):
        """Write a normal wig file"""
        file = tmpdir.join("file.txt")
        chrom = "test_chrom"
        reads = array([[1, 7], [0, 0], [0, 1], [1, 0]])
        write_wig(reads, file.strpath, chrom)

        assert file.read() == "\n".join(
            ["track type=wiggle_0", "variableStep chrom=test_chrom", "1\t7", "1\t0\n"]
        )

    def test_write_wig_empty(self, tmpdir):
        """try to write a wig file with no reads"""
        file = tmpdir.join("file.txt")
        chrom = ""
        reads = array([])

        with pytest.raises(ValueError) as e_info:
            write_wig(reads, file.strpath, chrom)

        assert e_info.value.args[0] == "requires non-empty reads"

    def test_write_wig_chr_empty(self, tmpdir):
        """try to write a wig file with no chromosome"""
        file = tmpdir.join("file.txt")
        chrom = ""
        reads = array([[1, 7], [0, 0], [0, 1], [1, 0]])

        write_wig(reads, file.strpath, chrom)

        assert file.read() == "\n".join(
            ["track type=wiggle_0", "variableStep chrom=", "1\t7", "1\t0\n"]
        )

    def test_open_wig(self, tmpdir):
        """open a normal wig file"""
        file = tmpdir.join("file.txt")
        with open(file, "w") as wigFH:
            wigFH.write("track type=wiggle_0\n")
            wigFH.write("variableStep chrom=test_chrom\n")
            wigFH.write("1\t5\n2\t6\n3\t8\n109\t1")

        reads, chrom = open_wig(file.strpath)

        assert chrom == "test_chrom"
        assert_array_equal(reads, array([[1, 5], [2, 6], [3, 8], [109, 1]]))

    def test_open_wig_noexist(self, tmpdir):
        """open a wig file that doesn't exist"""
        file = tmpdir.join("file.txt")

        with pytest.raises(FileNotFoundError):
            open_wig(file.strpath)

    def test_open_wig_empty(self, tmpdir):
        """open an empty wig file"""
        file = tmpdir.join("file.txt")
        with open(file, "w") as wigFH:
            wigFH.write("")

        with pytest.raises(ValueError) as e_info:
            open_wig(file.strpath)

        assert e_info.value.args[0][-10::] == "zero lines"

    def test_open_wig_malformatted(self, tmpdir):
        """open a funky wig file"""
        file = tmpdir.join("file.txt")
        with open(file, "w") as wigFH:
            wigFH.write("this is definitely\n")
            wigFH.write("not a proper wig file")

        with pytest.raises(ValueError) as e_info:
            open_wig(file.strpath)

        assert e_info.value.args[0] == "requires non-empty reads"

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
    """Mock making a new directory"""
    make_new_dir(["hello", "_rendseq", "_world"])
    mock_mkdir.assert_called_once_with("hello_rendseq_world")
