"""Build the optional C++ backtest engine as part of the Python package.

The pure-Python engine is the reference implementation and always works, so the
extension is genuinely optional: by default a build failure is reported as a
warning and the package installs without it (``runner_cpp`` then degrades to
``yabte_cpp = None`` and raises only if ``engine="cpp"`` is actually used).

``YABTE_BUILD_CPP`` controls that:

``auto`` (default)
    Try to build; on failure warn and carry on.
``1``
    Required -- any failure aborts the install. **Wheel builds must set this**,
    so a broken compile fails loudly instead of silently publishing a wheel
    with no extension in it.
``0``
    Skip the extension entirely.

``YABTE_CMAKE_TOOLCHAIN_FILE`` points at an existing Conan toolchain, letting a
caller run ``conan install`` once and reuse it across interpreter versions
instead of paying for dependency resolution per wheel.
"""

import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension

HERE = Path(__file__).parent.resolve()
CPP_DIR = HERE / "cpp"

MODULE_NAME = "yabte.backtest._yabte_backtest_lib"

_TRUTHY = {"1", "true", "yes", "on", "required"}
_FALSEY = {"0", "false", "no", "off", "skip"}


class CMakeExtension(Extension):
    """An extension built by CMake rather than by setuptools' compiler."""

    def __init__(self, name: str):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        mode = os.environ.get("YABTE_BUILD_CPP", "auto").strip().lower()

        if mode in _FALSEY:
            print("yabte: YABTE_BUILD_CPP is off, skipping the C++ engine")
            return

        required = mode in _TRUTHY
        try:
            for ext in self.extensions:
                self.build_cmake(ext)
        except Exception as exc:
            if required:
                raise
            warnings.warn(
                f"yabte: could not build the optional C++ engine ({exc}). "
                "Installing without it; the pure-Python engine is unaffected "
                'and engine="cpp" will raise if used. Set YABTE_BUILD_CPP=1 to '
                "turn this into an error.",
                stacklevel=2,
            )

    def build_cmake(self, ext: CMakeExtension):
        if not CPP_DIR.is_dir():
            raise RuntimeError(f"C++ sources not found at {CPP_DIR}")

        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError("cmake not found on PATH")

        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)
        self.discard_foreign_cache(build_temp)

        toolchain = os.environ.get("YABTE_CMAKE_TOOLCHAIN_FILE")
        if toolchain:
            toolchain_path = Path(toolchain).resolve()
            if not toolchain_path.is_file():
                raise RuntimeError(
                    f"YABTE_CMAKE_TOOLCHAIN_FILE={toolchain} does not exist"
                )
        else:
            toolchain_path = self.run_conan(build_temp)

        # The extension is installed into <build_lib>/yabte/backtest/ by the
        # install rule in cpp/pybind/CMakeLists.txt.
        install_prefix = Path(self.build_lib).resolve()
        install_prefix.mkdir(parents=True, exist_ok=True)

        self.spawn_checked(
            [
                cmake,
                "-S",
                str(CPP_DIR),
                "-B",
                str(build_temp),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DYABTE_WHEEL_ONLY=ON",
                f"-DCMAKE_TOOLCHAIN_FILE={toolchain_path}",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            ]
        )
        self.spawn_checked(
            [cmake, "--build", str(build_temp), "--config", "Release", "-j"]
        )
        self.spawn_checked([cmake, "--install", str(build_temp)])

        produced = sorted(install_prefix.glob("yabte/backtest/*.so"))
        if not produced:
            raise RuntimeError(
                f"CMake install produced no extension under {install_prefix}"
            )
        print(f"yabte: built {produced[0].relative_to(install_prefix)}")

    def run_conan(self, build_temp: Path) -> Path:
        conan = shutil.which("conan")
        if conan is None:
            raise RuntimeError(
                "conan not found on PATH and YABTE_CMAKE_TOOLCHAIN_FILE is unset"
            )

        self.spawn_checked([conan, "profile", "detect", "--exist-ok"])
        self.spawn_checked(
            [
                conan,
                "install",
                str(CPP_DIR),
                f"--output-folder={build_temp}",
                "--build=missing",
                "--settings=build_type=Release",
                "--options=wheel_only=True",
            ]
        )
        toolchain = build_temp / "conan_toolchain.cmake"
        if not toolchain.is_file():
            raise RuntimeError(f"conan did not generate {toolchain}")
        return toolchain

    @staticmethod
    def discard_foreign_cache(build_temp: Path) -> None:
        """Remove a CMake cache that was generated for a different source tree.

        CMake refuses to reuse a cache whose recorded paths do not match, and a
        stale one is easy to inherit: build directories travel with a copied
        project tree (cibuildwheel copies the whole project into its container)
        and they also pin the interpreter they were configured against.
        """
        cache = build_temp / "CMakeCache.txt"
        if not cache.is_file():
            return

        expected = str(CPP_DIR)
        try:
            recorded = next(
                line.split("=", 1)[1].strip()
                for line in cache.read_text(errors="replace").splitlines()
                if line.startswith("CMAKE_HOME_DIRECTORY:")
            )
        except (StopIteration, OSError):
            recorded = None

        if recorded == expected and str(build_temp) in cache.read_text(
            errors="replace"
        ):
            return

        print(
            f"yabte: discarding stale CMake cache in {build_temp} "
            f"(configured for {recorded!r}, need {expected!r})"
        )
        cache.unlink()
        shutil.rmtree(build_temp / "CMakeFiles", ignore_errors=True)

    @staticmethod
    def spawn_checked(cmd):
        print("yabte: " + " ".join(str(c) for c in cmd))
        subprocess.run(cmd, check=True)


setup(
    ext_modules=[CMakeExtension(MODULE_NAME)],
    cmdclass={"build_ext": CMakeBuild},
)
