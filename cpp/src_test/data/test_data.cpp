#include <stdexcept>
#include <string>

std::string test_data_path() {
    auto test_data_dir = std::getenv("YABTE_CPP_TEST_DATA_DIR");
    if (test_data_dir != nullptr)
        return test_data_dir;
    else
        return "/home/blair/projects/yabte_cpp/src_test/data";
    throw std::runtime_error("YABTE_CPP_TEST_DATA_DIR not set");
}