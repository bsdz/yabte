#pragma once

#include <arrow/api.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include <chrono>
#include <cmath>

using std::pow, std::round;

using Timestamp = std::chrono::system_clock::time_point;

inline Timestamp timestamp_from_ns(const arrow::TimestampType::c_type &ns) {
    return std::chrono::time_point<std::chrono::system_clock>(
        std::chrono::nanoseconds(ns));
}

/* ideally DayData would be a map like
DayData = map<string, variant<string, double>>
*/
using DayData = arrow::Table;

inline double round_n_digits(const double &value, const int &n) {
    return round(value * pow(10, n)) / pow(10, n);
}