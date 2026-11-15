"""Run auditwheel repair, excluding the Arrow libraries that pyarrow provides.

The extension deliberately links the Arrow shared libraries inside the installed pyarrow
wheel. auditwheel's default behaviour is to vendor every external library into the
wheel, which here would put a second Arrow runtime in the process alongside the one
pyarrow loads -- the "Two Arrows" problem. Those libraries must stay external, which is
what the pyarrow pin in [project.dependencies] guarantees.

The exclusions are read from the built extension's own DT_NEEDED entries rather than
hardcoded, because Arrow's sonames carry its version (libarrow.so.2300) and a stale list
would silently start vendoring again after a pyarrow bump. They are read from the wheel
instead of from an installed pyarrow because this runs in auditwheel's environment,
which has no pyarrow in it.

Usage: python scripts/repair_wheel.py <wheel> <dest_dir>
"""

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Any dependency whose soname starts with one of these is supplied by pyarrow.
PYARROW_LIB_PREFIXES = ("libarrow", "libparquet")


def needed_libraries(so_path: Path) -> list[str]:
    """Return the DT_NEEDED entries of an ELF shared object."""
    if shutil.which("patchelf"):
        # auditwheel requires patchelf, so it is always present here.
        out = subprocess.run(
            ["patchelf", "--print-needed", str(so_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]

    if shutil.which("readelf"):
        out = subprocess.run(
            ["readelf", "-d", str(so_path)], check=True, capture_output=True, text=True
        ).stdout
        names = []
        for line in out.splitlines():
            if "(NEEDED)" in line and "[" in line:
                names.append(line.split("[", 1)[1].split("]", 1)[0])
        return names

    sys.exit("repair_wheel: neither patchelf nor readelf available")


def arrow_exclusions(wheel: Path) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(wheel) as zf:
            members = [n for n in zf.namelist() if n.endswith(".so")]
            if not members:
                sys.exit(
                    f"repair_wheel: {wheel.name} contains no extension module. "
                    "The build produced a wheel without the C++ engine; "
                    "YABTE_BUILD_CPP must be 1 for wheel builds."
                )
            extracted = [Path(zf.extract(n, tmp)) for n in members]

        excludes: set[str] = set()
        for so in extracted:
            for name in needed_libraries(so):
                if name.startswith(PYARROW_LIB_PREFIXES):
                    excludes.add(name)

    if not excludes:
        sys.exit(
            "repair_wheel: the extension does not link any Arrow library, "
            "which means it was not built against pyarrow as expected"
        )
    return sorted(excludes)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <wheel> <dest_dir>")
    wheel, dest_dir = Path(sys.argv[1]), sys.argv[2]

    excludes = arrow_exclusions(wheel)
    print(f"repair_wheel: keeping external (pyarrow-provided): {', '.join(excludes)}")

    cmd = ["auditwheel", "repair", "-w", dest_dir, str(wheel)]
    for name in excludes:
        cmd += ["--exclude", name]

    print("repair_wheel: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
