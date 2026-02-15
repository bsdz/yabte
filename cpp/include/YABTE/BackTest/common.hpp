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

/* ideally DayData would be a map like
DayData = map<string, variant<string, double>>
*/
using DayData = arrow::Table;

inline double round_dp(const double &value, const optional<int> &dp = nullopt) {
    if (!dp.has_value()) {
        return value;
    }
    double multiplier = std::pow(10.0, dp.value());
    double x = value * multiplier;
    double fl = std::floor(x);
    double ce = std::ceil(x);

    double diff_fl = x - fl;
    double diff_ce = ce - x;

    double result;
    // TODO: use a small epsilon for float comparison to catch cases like
    // 6246.500000000000003
    if (diff_fl < diff_ce) {
        result = fl;
    } else if (diff_ce < diff_fl) {
        result = ce;
    } else {
        // It's exactly halfway (0.5), use the even rule
        result = (std::fmod(fl, 2.0) == 0) ? fl : ce;
    }

    return result / multiplier;
}
