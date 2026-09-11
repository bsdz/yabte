#include <gtest/gtest.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include "YABTE/Utilities/Arrow/TableHelpers.hpp"

#include <vector>
#include <cmath>

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
    
    // Should be all NaNs/Nulls
    // My implementation puts NaNs, which Arrow treats as value if not explicitly null, 
    // BUT I used AppendValues with NaNs. 
    // Wait, Arrow's DoubleBuilder doesn't automatically treat NaN as null unless validity bitmap is set.
    // In my code:
    // for(size_t i=0; i<data.size(); ++i) ma.push_back(std::nan(""));
    // ...
    // dbl_builder.AppendValues(ma.data(), ma.size())
    // So they will be valid doubles with value NaN.
    
    ASSERT_TRUE(std::isnan(double_array->Value(0)));
    ASSERT_TRUE(std::isnan(double_array->Value(1)));
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
    
    ASSERT_TRUE(std::isnan(double_array->Value(0))); // Index 0 (n-3)
    ASSERT_TRUE(std::isnan(double_array->Value(1))); // Index 1 (n-2)
    EXPECT_DOUBLE_EQ(double_array->Value(2), 2.0);   // Index 2 (n-1) -> Average
}
