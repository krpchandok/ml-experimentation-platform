#include <algorithm>

#include "check.h"
#include "process_tree.h"

using namespace mlplat;

namespace {

std::vector<int> sorted(std::vector<int> values) {
    std::sort(values.begin(), values.end());
    return values;
}

}

TEST_CASE("find_new_descendants follows multi-level chains") {
    std::vector<PidLink> candidates = {{11, 10}, {12, 11}, {13, 12}, {20, 1}, {21, 20}};
    auto adopted = sorted(find_new_descendants({10}, candidates));
    REQUIRE(adopted.size() == 3);
    CHECK_EQ(adopted[0], 11);
    CHECK_EQ(adopted[1], 12);
    CHECK_EQ(adopted[2], 13);
}

TEST_CASE("find_new_descendants handles candidates listed before their parents") {
    std::vector<PidLink> candidates = {{13, 12}, {12, 11}, {11, 10}};
    CHECK_EQ(find_new_descendants({10}, candidates).size(), 3u);
}

TEST_CASE("find_new_descendants attaches to any tracked ancestor") {
    std::vector<PidLink> candidates = {{30, 5}, {31, 7}, {32, 99}};
    auto adopted = sorted(find_new_descendants({5, 7}, candidates));
    REQUIRE(adopted.size() == 2);
    CHECK_EQ(adopted[0], 30);
    CHECK_EQ(adopted[1], 31);
}

TEST_CASE("find_new_descendants ignores unrelated processes and cycles") {
    CHECK(find_new_descendants({10}, {{20, 1}, {21, 20}}).empty());
    CHECK(find_new_descendants({10}, {}).empty());
    CHECK(find_new_descendants({10}, {{40, 41}, {41, 40}}).empty());
}
