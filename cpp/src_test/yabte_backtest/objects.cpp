#include <gtest/gtest.h>

#include "YABTE/BackTest/Asset.hpp"
#include "YABTE/BackTest/Book.hpp"
#include "YABTE/BackTest/Transaction.hpp"

using YABTE::BackTest::Book, YABTE::BackTest::CashTransaction, YABTE::BackTest::OHLCAsset,
    YABTE::BackTest::Trade, YABTE::BackTest::TransactionVector;

TEST(TransactionTest, BasicAssertions) {
    auto ts = timestamp_from_ns(0);
    auto t = Trade(ts, 100., 10., "asset", "order");
    EXPECT_NEAR(t.total_, -1000., 0.0001);
}

TEST(TransactionTest, CloneKeepsType) {
    auto ts = timestamp_from_ns(0);
    auto trade = Trade(ts, 100., 10., "asset", "order").clone();
    auto cash = CashTransaction(ts, 5., "interest").clone();
    ASSERT_NE(dynamic_cast<const Trade *>(trade.get()), nullptr);
    ASSERT_NE(dynamic_cast<const CashTransaction *>(cash.get()), nullptr);

    // a book only accepts trades and cash transactions, so a sliced clone throws
    auto book = Book("Main", "USD", 0., 0., 2);
    book.add_transactions(TransactionVector{trade, cash});
    EXPECT_NEAR(book.cash_, -995., 0.0001);
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
