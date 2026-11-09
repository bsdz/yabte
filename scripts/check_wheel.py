"""Acceptance check for a built yabte wheel, run by cibuildwheel's test step.

Importing the package is not enough of a test: the extension resolves Arrow at load time
through an rpath into the installed pyarrow, so a wheel can import fine and still fail
the moment the engine is used, or -- worse -- silently run against a second, vendored
copy of Arrow. This asserts the whole path works: the extension is present, it loads, it
runs, and it agrees with the pure-Python reference engine.

Exits non-zero with a diagnosis on any failure.
"""

import sys
from decimal import Decimal
from pathlib import Path


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_extension_present() -> Path:
    import yabte.backtest

    pkg_dir = Path(yabte.backtest.__file__).parent

    # Guard against testing the source tree instead of the installed wheel: if
    # the working directory is the repository root, "import yabte" resolves
    # there and this check would silently prove nothing.
    if "site-packages" not in str(pkg_dir):
        fail(
            f"imported yabte from {pkg_dir}, which is not an installed "
            "package. Run this from a directory outside the source tree."
        )
    found = sorted(pkg_dir.glob("_yabte_backtest_lib*.so"))
    if not found:
        fail(
            f"no extension module in {pkg_dir}. The wheel was built without the "
            "C++ engine (YABTE_BUILD_CPP must be 1 for wheel builds)."
        )
    print(f"ok: extension present: {found[0].name}")
    return found[0]


def check_no_vendored_arrow(so_path: Path) -> None:
    """The wheel must not carry its own Arrow; it must use pyarrow's."""
    for parent in so_path.parents:
        libs_dir = parent / "yabte.libs"
        if libs_dir.is_dir():
            vendored = [p.name for p in libs_dir.iterdir() if "arrow" in p.name.lower()]
            if vendored:
                fail(
                    f"wheel vendored Arrow libraries into {libs_dir}: {vendored}. "
                    "Two Arrow runtimes in one process will crash or corrupt "
                    "data; add them to auditwheel's --exclude list."
                )
    print("ok: no vendored Arrow libraries")


def check_loads_and_matches_pyarrow() -> None:
    import pyarrow

    import yabte.backtest.runner_cpp as rc

    if rc.yabte_cpp is None:
        fail(
            "runner_cpp could not import the extension, so it fell back to the "
            "pure-Python engine. Most likely Arrow failed to resolve at load "
            f"time (installed pyarrow is {pyarrow.__version__}); check the "
            "module's rpath and the pyarrow pin."
        )
    print(f"ok: extension loaded against pyarrow {pyarrow.__version__}")


def check_engines_agree() -> None:
    import pandas as pd

    from yabte.backtest import Book, OHLCAsset, SimpleOrder, Strategy, StrategyRunner

    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    prices = [100.0 + i for i in range(len(dates))]
    df = pd.DataFrame(
        [[p, p + 1.0, p - 1.0, p, 1000.0] for p in prices],
        columns=pd.MultiIndex.from_product(
            [["GOOG"], ["Open", "High", "Low", "Close", "Volume"]]
        ),
        index=dates,
    )

    class Buy(Strategy):
        def on_close(self):
            if len(self.data) == 3:
                self.orders.append(SimpleOrder(asset_name="GOOG", size=10))

    def run(engine):
        return StrategyRunner(
            data=df,
            assets=[OHLCAsset(name="GOOG", denom="USD")],
            strategies=[Buy()],
            books=[Book(name="Main", cash=Decimal("100000"))],
            engine=engine,
        ).run()

    res_py = run("python")
    res_cpp = run("cpp")

    tx_py, tx_cpp = res_py.transaction_history, res_cpp.transaction_history
    if len(tx_py) != len(tx_cpp):
        fail(f"transaction count differs: python={len(tx_py)} cpp={len(tx_cpp)}")
    if len(tx_py) == 0:
        fail("the smoke-test strategy traded nothing; the check proves nothing")

    hist_py = res_py.book_history
    hist_cpp = res_cpp.book_history
    if not hist_py.index.equals(hist_cpp.index):
        fail(
            "book history index differs between engines:\n"
            f"  python: {hist_py.index[0]} .. {hist_py.index[-1]} (n={len(hist_py)})\n"
            f"  cpp   : {hist_cpp.index[0]} .. {hist_cpp.index[-1]} (n={len(hist_cpp)})"
        )

    for col in hist_py.columns:
        a = hist_py[col].astype(float)
        b = hist_cpp[col].astype(float)
        worst = (a - b).abs().max()
        if worst > 1e-6:
            fail(f"book history column {col} differs by up to {worst}")

    for i in range(len(tx_py)):
        a, b = tx_py.iloc[i], tx_cpp.iloc[i]
        if a.ts != b.ts:
            fail(f"trade {i} timestamp differs: python={a.ts} cpp={b.ts}")
        if abs(float(a.quantity) - float(b.quantity)) > 1e-9:
            fail(f"trade {i} quantity differs: {a.quantity} vs {b.quantity}")
        if abs(float(a.price) - float(b.price)) > 1e-9:
            fail(f"trade {i} price differs: {a.price} vs {b.price}")

    print(f"ok: both engines agree ({len(tx_py)} trades, {len(hist_py)} history rows)")


def main() -> None:
    so = check_extension_present()
    check_no_vendored_arrow(so)
    check_loads_and_matches_pyarrow()
    check_engines_agree()
    print("wheel OK")


if __name__ == "__main__":
    main()
