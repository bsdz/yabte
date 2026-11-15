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

    def generate(self):
        """Collect the licences of everything linked into the build.

        Writes THIRD_PARTY_LICENSES.txt to the generators folder (next to
        conan_toolchain.cmake). setup.py ships it beside the compiled extension,
        since these libraries are compiled into it; build-only tools (e.g.
        CMake) are not linked and so are excluded.
        """
        sections = []
        for dep in sorted(
            self.dependencies.host.values(), key=lambda d: str(d.ref.name)
        ):
            lic = dep.license
            if isinstance(lic, (list, tuple)):
                lic = ", ".join(lic)
            header = f"{dep.ref.name} {dep.ref.version} (licence: {lic})"
            texts = []
            licenses_dir = (
                os.path.join(dep.package_folder, "licenses")
                if dep.package_folder
                else None
            )
            if licenses_dir and os.path.isdir(licenses_dir):
                for root, _, files in sorted(os.walk(licenses_dir)):
                    for name in sorted(files):
                        path = os.path.join(root, name)
                        with open(path, encoding="utf-8", errors="replace") as f:
                            rel = os.path.relpath(path, licenses_dir)
                            texts.append(f"--- {rel} ---\n{f.read().strip()}\n")
            if not texts:
                texts.append("(no licence file packaged by Conan)\n")
            sections.append("=" * 79 + f"\n{header}\n" + "=" * 79 + "\n\n")
            sections.append("\n".join(texts) + "\n")

        out = os.path.join(self.generators_folder, "THIRD_PARTY_LICENSES.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(
                "Third-party software compiled into the yabte C++ extension "
                "(_yabte_backtest_lib).\nyabte itself is MIT licensed; see "
                "LICENSE in the yabte distribution.\n\n"
            )
            f.writelines(sections)
