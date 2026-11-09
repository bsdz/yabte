#include <cstdlib>
#include <filesystem>
#include <string>

// Directory holding the test data files: YABTE_CPP_TEST_DATA_DIR if set, else the
// directory of this source file (src_test/data), so any target compiling it works.
std::string test_data_path() {
    if (auto dir = std::getenv("YABTE_CPP_TEST_DATA_DIR"); dir != nullptr) {
        return dir;
    }
    return std::filesystem::path(__FILE__).parent_path().string();
}
