from mypyc.build import mypycify

modules = [
    "yabte/backtest/__init__.py",
    "yabte/backtest/_helpers.py",
    "yabte/backtest/asset.py",
    "yabte/backtest/book.py",
    "yabte/backtest/order.py",
    "yabte/backtest/strategy.py",
    "yabte/backtest/transaction.py",
]

extensions = mypycify(modules)


def build(setup_kwargs):
    """Needed for the poetry building interface."""

    setup_kwargs.update(
        {
            "ext_modules": extensions,
        }
    )
