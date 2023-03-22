import unittest

from yabte.tests._helpers import notebooks_dir

HAS_NBFORMAT = True
try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except:
    HAS_NBFORMAT = False


class NotebooksTestCase(unittest.TestCase):
    @unittest.skipUnless(HAS_NBFORMAT, "needs nbformat")
    def test_notebooks_smoke(self):
        for nb in notebooks_dir.glob("*.ipynb"):
            if not nb.name.startswith("_"):
                with self.subTest(nb.name):
                    ep = ExecutePreprocessor(timeout=600)
                    with nb.open() as f:
                        nb = nbformat.read(f, as_version=4)
                        # will raise exception if issue, eg missing module
                        ep.preprocess(nb, {"metadata": {"path": str(notebooks_dir)}})
