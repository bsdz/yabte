#include <cstdlib>
#include <string>

// Directory holding the test data files. YABTE_CPP_TEST_DATA_DIR overrides the
// default, which CMake sets to this source directory (src_test/data).
std::string test_data_path() {
    if (auto dir = std::getenv("YABTE_CPP_TEST_DATA_DIR"); dir != nullptr) {
        return dir;
    }
    return YABTE_CPP_TEST_DATA_DIR_DEFAULT;
}
