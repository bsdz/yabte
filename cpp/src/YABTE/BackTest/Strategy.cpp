#include "YABTE/BackTest/Strategy.hpp"

namespace YABTE::BackTest {

template <class Var, class>
std::ostream& operator<<(std::ostream& os, Var const& v) {
    std::visit([&os](auto const& x) { os << x; }, v);
    return os;
}

std::ostream& operator<<(std::ostream& os, ParamMap const& v) {
    for (auto i : v) os << i.first << ": " << i.second << ", ";
    return os;
}

bool Strategy::operator==(Strategy const& rhs) const {
    return this == std::addressof(rhs);
}

void Strategy::init() {}
void Strategy::on_open() {}
void Strategy::on_close() {}

shared_ptr<const Table> Strategy::extend_data(
    const shared_ptr<const Table>& data) {
    return nullptr;
}

}  // namespace YABTE::BackTest