#include <arrow/api.h>
#include <gtest/gtest.h>

#include <cmath>

#include "YABTE/BackTest/Asset.hpp"
#include "YABTE/BackTest/Book.hpp"
#include "YABTE/BackTest/Transaction.hpp"

using YABTE::BackTest::Book, YABTE::BackTest::CashTransaction, YABTE::BackTest::CostType, YABTE::BackTest::FillPrice,
    YABTE::BackTest::MissingPriceError, YABTE::BackTest::OHLCAsset,
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

namespace {
// one row per element; NaN becomes a null, as from pandas
std::shared_ptr<arrow::Table> ohlc_table(std::vector<double> open,
                                         std::vector<double> high,
                                         std::vector<double> low,
                                         std::vector<double> close) {
    arrow::FieldVector fields;
    arrow::ArrayVector arrays;
    auto add = [&](const std::string &name, const std::vector<double> &values) {
        arrow::DoubleBuilder b;
        for (auto v : values) {
            EXPECT_TRUE((std::isnan(v) ? b.AppendNull() : b.Append(v)).ok());
        }
        fields.push_back(arrow::field(name, arrow::float64()));
        arrays.push_back(b.Finish().ValueOrDie());
    };
    add("foo.Open", open);
    add("foo.High", high);
    add("foo.Low", low);
    add("foo.Close", close);
    return arrow::Table::Make(arrow::schema(fields), arrays);
}
}  // namespace

TEST(AssetTest, FillPrice) {
    auto nan = std::nan("");
    // row 0 has every price; row 1 has no High or Open
    auto t = ohlc_table({10., nan}, {14., nan}, {11., 11.}, {13., 13.});
    auto price = [&](FillPrice fp, int64_t row) {
        return OHLCAsset("foo", "USD", 2, 2, "foo", fp).intraday_traded_price(t, row);
    };
    EXPECT_NEAR(price(FillPrice::MIDPOINT, 0), 12.5, 1e-9);
    EXPECT_NEAR(price(FillPrice::MIDPOINT, 1), 13., 1e-9);  // falls back to Close
    EXPECT_NEAR(price(FillPrice::OPEN, 0), 10., 1e-9);
    EXPECT_TRUE(std::isnan(price(FillPrice::OPEN, 1)));  // no fallback
    EXPECT_NEAR(price(FillPrice::CLOSE, 0), 13., 1e-9);

    // the setting survives the clone each run makes
    auto a = OHLCAsset("foo", "USD", 2, 2, "foo", FillPrice::OPEN);
    auto c = std::dynamic_pointer_cast<OHLCAsset>(a.clone());
    EXPECT_EQ(c->fill_price_, FillPrice::OPEN);
}

TEST(AssetTest, TransactionCost) {
    auto cost = [](CostType ct, double c, double q, double p) {
        return OHLCAsset("foo", "USD", 2, 2, "foo", FillPrice::MIDPOINT, c, ct)
            .transaction_cost(q, p);
    };
    EXPECT_NEAR(cost(CostType::BPS, 10., -5., 100.), 0.5, 1e-12);
    EXPECT_NEAR(cost(CostType::PER_UNIT, 0.65, -2., 100.), 1.3, 1e-12);
    EXPECT_NEAR(cost(CostType::PER_ORDER, 1., 7., 100.), 1., 1e-12);
    EXPECT_EQ(cost(CostType::PER_ORDER, 1., 0., 100.), 0.);  // nothing traded
    EXPECT_EQ(cost(CostType::BPS, 0., 5., 100.), 0.);         // the default
    EXPECT_THROW(cost(CostType::BPS, -1., 5., 100.), std::invalid_argument);

    // the setting survives the clone each run makes
    auto a = OHLCAsset("foo", "USD", 2, 2, "foo", FillPrice::MIDPOINT, 3.,
                       CostType::PER_UNIT);
    auto c = a.clone();
    EXPECT_EQ(c->cost_, 3.);
    EXPECT_EQ(c->cost_type_, CostType::PER_UNIT);
}

TEST(AssetTest, DataCost) {
    // one row: foo.HalfSpread = 0.05, foo.Bad = -1, foo.Gap = null
    arrow::DoubleBuilder spread, bad, gap;
    ASSERT_TRUE(spread.Append(0.05).ok());
    ASSERT_TRUE(bad.Append(-1.).ok());
    ASSERT_TRUE(gap.AppendNull().ok());
    auto schema = arrow::schema({arrow::field("foo.HalfSpread", arrow::float64()),
                                 arrow::field("foo.Bad", arrow::float64()),
                                 arrow::field("foo.Gap", arrow::float64())});
    auto table = arrow::Table::Make(schema, {spread.Finish().ValueOrDie(),
                                             bad.Finish().ValueOrDie(),
                                             gap.Finish().ValueOrDie()});
    auto ts = timestamp_from_ns(0);
    auto asset = [](const char *field, CostType ct) {
        return OHLCAsset("foo", "USD", 2, 2, "foo", FillPrice::MIDPOINT, 0.,
                         CostType::BPS, field, ct);
    };
    EXPECT_NEAR(asset("HalfSpread", CostType::PER_UNIT)
                    .data_cost(-4., 100., *table, 0, ts),
                0.2, 1e-12);
    EXPECT_NEAR(asset("HalfSpread", CostType::BPS)
                    .data_cost(-4., 100., *table, 0, ts),
                0.002, 1e-12);
    EXPECT_NEAR(asset("HalfSpread", CostType::PER_ORDER)
                    .data_cost(-4., 100., *table, 0, ts),
                0.05, 1e-12);
    // nothing traded: the value isn't read, so gaps and bad values don't matter
    EXPECT_EQ(asset("Gap", CostType::PER_UNIT).data_cost(0., 100., *table, 0, ts),
              0.);
    EXPECT_THROW(
        asset("Gap", CostType::PER_UNIT).data_cost(1., 100., *table, 0, ts),
        MissingPriceError);
    EXPECT_THROW(
        asset("Missing", CostType::PER_UNIT).data_cost(1., 100., *table, 0, ts),
        MissingPriceError);
    EXPECT_THROW(
        asset("Bad", CostType::PER_UNIT).data_cost(1., 100., *table, 0, ts),
        std::invalid_argument);
    // no cost_field
    EXPECT_EQ(OHLCAsset("foo", "USD").data_cost(1., 100., *table, 0, ts), 0.);
    // the setting survives the clone each run makes
    auto c = asset("HalfSpread", CostType::BPS).clone();
    EXPECT_EQ(c->cost_field_, "HalfSpread");
    EXPECT_EQ(c->cost_field_type_, CostType::BPS);
}
