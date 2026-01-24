#include <gtest/gtest.h>

#include "YABTE/BackTest/Asset.hpp"
#include "YABTE/BackTest/Transaction.hpp"

using YABTE::BackTest::OHLCAsset, YABTE::BackTest::Trade;

TEST(TransactionTest, BasicAssertions) {
    auto ts = timestamp_from_ns(0);
    auto t = Trade(ts, 100., 10., "asset", "order");
    EXPECT_NEAR(t.total_, -1000., 0.0001);
}

TEST(AssetTest, BasicAssertions) {
    auto a = OHLCAsset("foo", "USD");
    EXPECT_EQ(a.name_, "foo");
    EXPECT_EQ(a.denom_, "USD");
    EXPECT_EQ(a.price_round_dp_, 2);
    EXPECT_EQ(a.quantity_round_dp_, 2);
    EXPECT_EQ(a.data_label_, a.name_);
    EXPECT_NEAR(a.round_quantity(1.2345), 1.23, 0.0001);
}
