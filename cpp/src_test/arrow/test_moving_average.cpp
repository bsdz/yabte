#include <gtest/gtest.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include "YABTE/Utilities/Arrow/TableHelpers.hpp"

#include <cmath>
#include <vector>

using namespace YABTE::Utilities::Arrow;

TEST(ArrowTest, ComputeMovingAverage_Basic) {
    // Create input array: [1, 2, 3, 4, 5]
    arrow::DoubleBuilder builder;
    ASSERT_TRUE(builder.Append(1.0).ok());
    ASSERT_TRUE(builder.Append(2.0).ok());
    ASSERT_TRUE(builder.Append(3.0).ok());
    ASSERT_TRUE(builder.Append(4.0).ok());
    ASSERT_TRUE(builder.Append(5.0).ok());
    
    std::shared_ptr<arrow::Array> array;
    ASSERT_TRUE(builder.Finish(&array).ok());
    auto chunked_array = std::make_shared<arrow::ChunkedArray>(array);

    // Compute MA with window=3
    // Expected: [NaN, NaN, (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4]
    auto res = ComputeMovingAverage(chunked_array, 3);
    ASSERT_TRUE(res.ok());
    auto ma_array = res.ValueOrDie();

    ASSERT_EQ(ma_array->length(), 5);
    
    // Check values
    // chunked array might have multiple chunks, but here likely 1
    ASSERT_EQ(ma_array->num_chunks(), 1);
    auto double_array = std::dynamic_pointer_cast<arrow::DoubleArray>(ma_array->chunk(0));
    
    ASSERT_TRUE(double_array->IsNull(0));
    ASSERT_TRUE(double_array->IsNull(1));
    EXPECT_DOUBLE_EQ(double_array->Value(2), 2.0);
    EXPECT_DOUBLE_EQ(double_array->Value(3), 3.0);
    EXPECT_DOUBLE_EQ(double_array->Value(4), 4.0);
}

TEST(ArrowTest, ComputeMovingAverage_InsufficientData) {
    // Input: [1, 2], Window: 3
    arrow::DoubleBuilder builder;
    ASSERT_TRUE(builder.Append(1.0).ok());
    ASSERT_TRUE(builder.Append(2.0).ok());
    
    std::shared_ptr<arrow::Array> array;
    ASSERT_TRUE(builder.Finish(&array).ok());
    auto chunked_array = std::make_shared<arrow::ChunkedArray>(array);

    auto res = ComputeMovingAverage(chunked_array, 3);
    ASSERT_TRUE(res.ok());
    auto ma_array = res.ValueOrDie();

    ASSERT_EQ(ma_array->length(), 2);
    auto double_array = std::dynamic_pointer_cast<arrow::DoubleArray>(ma_array->chunk(0));

    // Fewer rows than the window: every slot is in the warmup region, which
    // the implementation represents as null (not a NaN value).
    ASSERT_EQ(double_array->null_count(), 2);
    ASSERT_TRUE(double_array->IsNull(0));
    ASSERT_TRUE(double_array->IsNull(1));
}

TEST(ArrowTest, ComputeMovingAverage_ExactSize) {
    // Input: [1, 2, 3], Window: 3
    arrow::DoubleBuilder builder;
    ASSERT_TRUE(builder.Append(1.0).ok());
    ASSERT_TRUE(builder.Append(2.0).ok());
    ASSERT_TRUE(builder.Append(3.0).ok());
    
    std::shared_ptr<arrow::Array> array;
    ASSERT_TRUE(builder.Finish(&array).ok());
    auto chunked_array = std::make_shared<arrow::ChunkedArray>(array);

    auto res = ComputeMovingAverage(chunked_array, 3);
    ASSERT_TRUE(res.ok());
    auto ma_array = res.ValueOrDie();

    ASSERT_EQ(ma_array->length(), 3);
    auto double_array = std::dynamic_pointer_cast<arrow::DoubleArray>(ma_array->chunk(0));
    
    ASSERT_TRUE(double_array->IsNull(0));           // warmup
    ASSERT_TRUE(double_array->IsNull(1));           // warmup
    ASSERT_FALSE(double_array->IsNull(2));
    EXPECT_DOUBLE_EQ(double_array->Value(2), 2.0);  // (1+2+3)/3
}

TEST(ArrowTest, ComputeMovingAverage_GapOnlyAffectsItsWindows) {
    // Input: [1, 2, null, 4, 5, 6, NaN, 8, 9, 10], Window: 2
    // As pandas rolling(2).mean(): only the windows containing a gap are null.
    arrow::DoubleBuilder builder;
    for (double v : {1.0, 2.0}) ASSERT_TRUE(builder.Append(v).ok());
    ASSERT_TRUE(builder.AppendNull().ok());
    for (double v : {4.0, 5.0, 6.0, std::nan(""), 8.0, 9.0, 10.0})
        ASSERT_TRUE(builder.Append(v).ok());

    std::shared_ptr<arrow::Array> array;
    ASSERT_TRUE(builder.Finish(&array).ok());
    auto res = ComputeMovingAverage(std::make_shared<arrow::ChunkedArray>(array), 2);
    ASSERT_TRUE(res.ok());
    auto ma = std::dynamic_pointer_cast<arrow::DoubleArray>(res.ValueOrDie()->chunk(0));
    ASSERT_EQ(ma->length(), 10);

    const std::vector<double> expected = {0, 1.5, 0, 0, 4.5, 5.5, 0, 0, 8.5, 9.5};
    const std::vector<bool> is_null = {true, false, true, true, false,
                                       false, true, true, false, false};
    for (int64_t i = 0; i < ma->length(); ++i) {
        EXPECT_EQ(ma->IsNull(i), is_null[i]) << "row " << i;
        if (!is_null[i]) EXPECT_DOUBLE_EQ(ma->Value(i), expected[i]) << "row " << i;
    }
}
