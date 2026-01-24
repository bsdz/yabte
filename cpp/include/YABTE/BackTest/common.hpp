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
    if (n == 0) return std::round(value);
    
    // Python 3's round() uses "round half to even"
    double p = std::pow(10.0, n);
    double val_scaled = value * p;
    
    // Check if exactly half (e.g. 0.5, 1.5, 2.5)
    // To do this robustly with floating point, check if difference from nearest integer is 0.5
    double floor_val = std::floor(val_scaled);
    double diff = val_scaled - floor_val;
    
    // Using a small epsilon for float comparison
    if (std::abs(diff - 0.5) < 1e-15) {
        // Round to nearest even integer
        // If floor is even, round down (keep floor). If floor is odd, round up (floor + 1)
        if (std::fmod(floor_val, 2.0) == 0.0) {
            return floor_val / p;
        } else {
            return (floor_val + 1.0) / p;
        }
    }
    
    return std::round(val_scaled) / p;
}