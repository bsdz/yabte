import os

from conan import ConanFile
from conan.tools.cmake import cmake_layout
from conan.tools.env import Environment


class YabteProject(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain", "VirtualRunEnv", "VirtualBuildEnv"

    def requirements(self):
        self.requires("arrow/23.0.0")
        self.requires("glog/0.7.1")
        self.requires("gtest/1.17.0")
        self.requires("pybind11/3.0.1")
        self.requires("bshoshany-thread-pool/5.1.0")
        self.requires("re2/20251105")

    def configure(self):
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
