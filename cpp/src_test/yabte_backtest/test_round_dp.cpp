#include <gtest/gtest.h>

#include "YABTE/BackTest/common.hpp"

// round_dp must match the Python engine, which rounds the exact value of the
// double (Decimal(float)) half-to-even. Expected values below were produced by
// `round(Decimal(x), dp)` in Python.

TEST(RoundDpTest, NoRoundingWhenDpUnset) {
    EXPECT_DOUBLE_EQ(round_dp(103.465), 103.465);
    EXPECT_DOUBLE_EQ(round_dp(103.465, std::nullopt), 103.465);
}

TEST(RoundDpTest, PlainCases) {
    EXPECT_DOUBLE_EQ(round_dp(1.234, 2), 1.23);
    EXPECT_DOUBLE_EQ(round_dp(1.236, 2), 1.24);
    EXPECT_DOUBLE_EQ(round_dp(-1.234, 2), -1.23);
    EXPECT_DOUBLE_EQ(round_dp(-1.236, 2), -1.24);
    EXPECT_DOUBLE_EQ(round_dp(100.0, 2), 100.0);
    EXPECT_DOUBLE_EQ(round_dp(0.0, 2), 0.0);
}

TEST(RoundDpTest, NearTieDecidedByExactValueOfDouble) {
    // The double nearest 103.465 is 103.46500000000000341..., i.e. *above* the
    // halfway point, even though 103.465 * 100 == 10346.5 exactly in binary.
    // Python: round(Decimal(103.465), 2) == Decimal('103.47').
    EXPECT_DOUBLE_EQ(round_dp(103.465, 2), 103.47);
    EXPECT_DOUBLE_EQ(round_dp(-103.465, 2), -103.47);

    // The double nearest 2.675 is 2.67499999999999982236431605997495353221893310546875,
    // i.e. *below* halfway. Python: round(Decimal(2.675), 2) == Decimal('2.67').
    EXPECT_DOUBLE_EQ(round_dp(2.675, 2), 2.67);
    EXPECT_DOUBLE_EQ(round_dp(-2.675, 2), -2.67);
}

TEST(RoundDpTest, ExactTieRoundsHalfToEven) {
    // Dyadic rationals are exact in binary, so these are true ties.
    EXPECT_DOUBLE_EQ(round_dp(0.5, 0), 0.0);
    EXPECT_DOUBLE_EQ(round_dp(1.5, 0), 2.0);
    EXPECT_DOUBLE_EQ(round_dp(2.5, 0), 2.0);
    EXPECT_DOUBLE_EQ(round_dp(-0.5, 0), 0.0);
    EXPECT_DOUBLE_EQ(round_dp(-1.5, 0), -2.0);
    EXPECT_DOUBLE_EQ(round_dp(-2.5, 0), -2.0);
    EXPECT_DOUBLE_EQ(round_dp(0.125, 2), 0.12);
    EXPECT_DOUBLE_EQ(round_dp(0.375, 2), 0.38);
}
