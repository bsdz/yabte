#pragma once

#include <arrow/api.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include <chrono>
#include <cmath>
#include <optional>

using std::pow, std::round, std::optional, std::nullopt;

using Timestamp = std::chrono::system_clock::time_point;

inline Timestamp timestamp_from_ns(const arrow::TimestampType::c_type &ns) {
    return std::chrono::time_point<std::chrono::system_clock>(
        std::chrono::nanoseconds(ns));
}

// Arrow timestamp columns carry their own resolution, and pandas picks one
// based on version (>=3 defaults DatetimeIndex to us, <3 used ns). Scale the
// raw value by the column's unit rather than assuming nanoseconds.
inline Timestamp timestamp_from_unit(const arrow::TimestampType::c_type &value,
                                     const arrow::TimeUnit::type &unit) {
    switch (unit) {
        case arrow::TimeUnit::SECOND:
            return std::chrono::time_point<std::chrono::system_clock>(
                std::chrono::duration_cast<std::chrono::system_clock::duration>(
                    std::chrono::seconds(value)));
        case arrow::TimeUnit::MILLI:
            return std::chrono::time_point<std::chrono::system_clock>(
                std::chrono::duration_cast<std::chrono::system_clock::duration>(
                    std::chrono::milliseconds(value)));
        case arrow::TimeUnit::MICRO:
            return std::chrono::time_point<std::chrono::system_clock>(
                std::chrono::duration_cast<std::chrono::system_clock::duration>(
                    std::chrono::microseconds(value)));
        case arrow::TimeUnit::NANO:
        default:
            return timestamp_from_ns(value);
    }
}

/* ideally DayData would be a map like
DayData = map<string, variant<string, double>>
*/
using DayData = arrow::Table;

inline double round_dp(const double &value, const optional<int> &dp = nullopt) {
    if (!dp.has_value()) {
        return value;
    }
    const double multiplier = std::pow(10.0, dp.value());

    // Mirror the Python engine, which rounds the *exact* value of the double
    // (Decimal(float) is exact) half-to-even. Multiplying by 10^dp can round
    // away the sub-ulp part that decides a near-tie (e.g. the double nearest
    // 103.465 is 103.46500000000000341..., yet 103.465 * 100 == 10346.5
    // exactly), so recover it: value * multiplier == x + err exactly, where
    // err is the FMA residual.
    const double x = value * multiplier;
    const double err = std::fma(value, multiplier, -x);

    const double fl = std::floor(x);
    const double ce = std::ceil(x);
    if (fl == ce) {
        return fl / multiplier;  // integral; |err| is far below a half
    }

    // (x - fl) is exact; d is a nonzero multiple of ulp(x) whenever it is
    // nonzero, and |err| <= ulp(x)/2, so d alone decides unless it is zero.
    const double d = (x - fl) - 0.5;

    double result;
    if (d > 0 || (d == 0 && err > 0)) {
        result = ce;
    } else if (d < 0 || (d == 0 && err < 0)) {
        result = fl;
    } else {
        // Exact tie: round half to even.
        result = (std::fmod(fl, 2.0) == 0) ? fl : ce;
    }
    return result / multiplier;
}
