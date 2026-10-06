#pragma once

#include <unordered_set>
#include <vector>

namespace mlplat {

struct PidLink {
    int pid = 0;
    int ppid = 0;
};

std::vector<int> find_new_descendants(const std::unordered_set<int>& tracked,
                                      const std::vector<PidLink>& candidates);

}
