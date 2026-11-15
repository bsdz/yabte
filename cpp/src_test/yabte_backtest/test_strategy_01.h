#pragma once

#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/table.h>

#include <cmath>
#include <memory>

#include "YABTE/BackTest/Strategy.hpp"

using YABTE::BackTest::Strategy;

using arrow::Table;

using std::shared_ptr;

shared_ptr<Table> MyExtendTable(const shared_ptr<const Table>& table, int n,
                                int m);

class TestSMAXOStrat : public Strategy {
   public:
    TestSMAXOStrat();

    shared_ptr<Strategy> clone() const override;
    shared_ptr<const Table> extend_data(
        const shared_ptr<const Table>& data) override;

    void on_open() override;
    void on_close() override;
};
