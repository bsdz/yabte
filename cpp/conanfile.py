import os

from conan import ConanFile
from conan.tools.cmake import cmake_layout
from conan.tools.env import Environment


class YabteProject(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain", "VirtualRunEnv", "VirtualBuildEnv"

    # wheel_only builds just the Python extension module. That links Arrow from
    # the installed pyarrow wheel (see the "Two Arrows" note in AGENTS.md) and
    # uses pyarrow's bundled headers, so Conan's Arrow -- and its Boost, Thrift,
    # OpenSSL and Snappy chain -- plus GoogleTest are all unnecessary. Omitting
    # them is what makes building wheels in CI practical.
    options = {"wheel_only": [True, False]}
    default_options = {"wheel_only": False}

    def requirements(self):
        self.requires("glog/0.7.1")
        self.requires("pybind11/3.0.1")
        self.requires("bshoshany-thread-pool/5.1.0")
        self.requires("re2/20251105")
        if not self.options.wheel_only:
            self.requires("arrow/23.0.0")
            self.requires("gtest/1.17.0")

    def configure(self):
        if self.options.wheel_only:
            return

        # Setting options for Arrow
        self.options["arrow"].filesystem_layer = True
        self.options["arrow"].parquet = True
        self.options["arrow"].with_thrift = True
        self.options["arrow"].with_snappy = True
        self.options["arrow"].with_re2 = True
        self.options["arrow"].compute = True
        self.options["arrow"].with_json = True

        # Setting options for Boost (Arrow dependency)
        self.options["boost"].without_cobalt = True
        self.options["boost"].without_test = True
