import unittest

import numpy as np

from yabte.tests._unittest_numpy_extensions import NumpyTestCase
from yabte.utilities.black_scholes import black_scholes


class BlackScholesTestCase(NumpyTestCase):
    def test_textbook_values(self):
        # Hull's standard example: S=K=100, one year, 20% vol, 5% rate
        call = black_scholes(100, 100, 1, 0.2, r=0.05)
        put = black_scholes(100, 100, 1, 0.2, r=0.05, call=False)
        self.numpyAssertAllclose(call.price, 10.450584, atol=1e-6)
        self.numpyAssertAllclose(put.price, 5.573526, atol=1e-6)
        self.numpyAssertAllclose(call.delta, 0.636831, atol=1e-6)
        self.numpyAssertAllclose(put.delta, call.delta - 1, atol=1e-12)

    def test_put_call_parity(self):
        S = np.linspace(50, 150, 11)
        K, tau, sigma, r = 100, 0.5, 0.3, 0.02
        call = black_scholes(S, K, tau, sigma, r)
        put = black_scholes(S, K, tau, sigma, r, call=False)
        self.numpyAssertAllclose(
            call.price - put.price, S - K * np.exp(-r * tau), atol=1e-10
        )
        self.numpyAssertAllclose(call.gamma, put.gamma, atol=1e-12)
        self.numpyAssertAllclose(call.vega, put.vega, atol=1e-12)

    def test_greeks_match_finite_differences(self):
        S, K, tau, sigma, r = 105.0, 100.0, 0.25, 0.25, 0.03
        h = 1e-4
        for call in (True, False):
            with self.subTest(call=call):
                bs = black_scholes(S, K, tau, sigma, r, call)

                def price(**kw):
                    args = dict(S=S, K=K, tau=tau, sigma=sigma, r=r, call=call)
                    return black_scholes(**{**args, **kw}).price

                self.numpyAssertAllclose(
                    bs.delta, (price(S=S + h) - price(S=S - h)) / (2 * h), atol=1e-6
                )
                self.numpyAssertAllclose(
                    bs.gamma,
                    (price(S=S + h) - 2 * bs.price + price(S=S - h)) / h**2,
                    atol=1e-4,
                )
                self.numpyAssertAllclose(
                    bs.vega,
                    (price(sigma=sigma + h) - price(sigma=sigma - h)) / (2 * h),
                    atol=1e-5,
                )

    def test_expiry_and_zero_vol(self):
        S = np.array([90.0, 100.0, 110.0])
        call = black_scholes(S, 100, 0, 0.2)
        put = black_scholes(S, 100, 0, 0.2, call=False)
        self.numpyAssertAllclose(call.price, [0, 0, 10])
        self.numpyAssertAllclose(put.price, [10, 0, 0])
        self.numpyAssertAllclose(call.delta, [0, 0, 1])
        self.numpyAssertAllclose(put.delta, [-1, 0, 0])
        self.numpyAssertAllclose(call.gamma, [0, 0, 0])
        self.numpyAssertAllclose(call.vega, [0, 0, 0])
        # zero vol: the discounted forward's intrinsic value
        flat = black_scholes(100, 100, 1, 0, r=0.05)
        self.numpyAssertAllclose(flat.price, 100 - 100 * np.exp(-0.05))
        self.numpyAssertAllclose(flat.delta, 1)


if __name__ == "__main__":
    unittest.main()
