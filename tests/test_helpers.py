import pytest
import numpy as np

from src.pypcfd import fault_diagnosis as fd

class TestStdBasisVectors:

    @pytest.mark.parametrize('shape, exp_shape',
                             [('col', (3, 1)),
                              ('row', (1, 3)),
                              ('flat', (3, ))])
    def test_row_vector(self, shape, exp_shape):
        e = fd.std_basis_vector(3, 2, shape)
        assert e.shape == exp_shape
