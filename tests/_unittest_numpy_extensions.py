import unittest

import numpy.testing as nptu

__all__ = ["NumpyTestCase"]


class NumpyTestCase(unittest.TestCase):
    """Specialized TestCase which includes numpy test assertion functions and
    maps them from assert_func_name to self.numpyAssertFuncName.

    class NumpyTest(NumpyTestCase):
    def test_allclose_example(self):
        a1 = np.array([1.,2.,3.])
        self.numpyAssertAllclose(a1, np.array([1.,2.,3.1]))
    """


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
        setattr(NumpyTestCase, new_name, make_test_wrapper(fn))


if __name__ == "__main__":
    import numpy as np

    class _NumpyTest(NumpyTestCase):
        def test_allclose_example(self):
            self.numpyAssertAllclose(
                np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.1])
            )

    # should fail!
    suite = unittest.TestSuite()
    suite.addTest(_NumpyTest("test_allclose_example"))
    unittest.TextTestRunner().run(suite)
