"""monkey patch unittest to include numpy assertions. Maps numpy
assert_func_name to numpyAssertFuncName.

Example:

    class NumpyWrapperTest(unittest.TestCase):
        def test_allclose_example(self):
            a1 = np.array([1.,2.,3.])
            self.numpyAssertAllclose(a1, np.array([1.,2.,3.1]))

    suite = unittest.TestSuite()
    suite.addTest(NumpyWrapperTest("test_allclose_example"))
    unittest.TextTestRunner().run(suite)
"""

import unittest

import numpy.testing as nptu


def make_test_wrapper(fn):
    def test_wrapper(self, *args, **kwargs):
        try:
            nptu.__dict__[fn](*args, **kwargs)
        except AssertionError as err:
            self.fail(err)

    return test_wrapper


for fn in nptu.__dict__:
    if fn.startswith("assert") and not fn.endswith("_"):
        new_name = "numpy" + fn.title().replace("_", "")
        setattr(unittest.TestCase, new_name, make_test_wrapper(fn))
