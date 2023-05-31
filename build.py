import sys

mypyc_modules = [
    "yabte/backtest/__init__.py",
    "yabte/backtest/_helpers.py",
    "yabte/backtest/asset.py",
    "yabte/backtest/book.py",
    "yabte/backtest/order.py",
    "yabte/backtest/strategy.py",
    "yabte/backtest/transaction.py",
]


def build(setup_kwargs):
    """Imported by poetry generated setup.py during build."""
    try:
        from mypyc.build import mypycify

        setup_kwargs.update(
            {
                "ext_modules": mypycify(mypyc_modules),
            }
        )
    except:
        pass


def build_inplace():
    """Build modules inplace for running tests"""
    import os
    import subprocess

    # mypyc generates a setup.py and calls build_ext --inplace
    env = os.environ.copy()
    cmd = subprocess.run([sys.executable, "-m", "mypyc"] + mypyc_modules, env=env)
    return cmd.returncode


def cleanup():
    from pathlib import Path

    this_dir = Path(__file__).parent

    # rm module shared objs
    for mod in mypyc_modules:
        fp = this_dir / mod
        for gfp in fp.parent.glob(fp.with_suffix("").name + "*.so"):
            gfp.unlink(missing_ok=True)

    for gfp in this_dir.glob("*__mypyc.*.so"):
        gfp.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if "clean" in sys.argv[1]:
            sys.exit(cleanup())
        elif "inplace" in sys.argv[1]:
            sys.exit(build_inplace())

    sys.exit("Unknown option, use '--clean' or '--inplace'")
