#include "process_tree.h"

#include <unordered_map>

namespace mlplat {

std::vector<int> find_new_descendants(const std::unordered_set<int>& tracked,
                                      const std::vector<PidLink>& candidates) {
    std::unordered_map<int, std::vector<int>> children_of;
    for (const PidLink& link : candidates) children_of[link.ppid].push_back(link.pid);

    std::vector<int> adopted;
    std::vector<int> frontier(tracked.begin(), tracked.end());
    std::unordered_set<int> visited(tracked.begin(), tracked.end());
    while (!frontier.empty()) {
        int parent = frontier.back();
        frontier.pop_back();
        auto found = children_of.find(parent);
        if (found == children_of.end()) continue;
        for (int child : found->second) {
            if (!visited.insert(child).second) continue;
            adopted.push_back(child);
            frontier.push_back(child);
        }
    }
    return adopted;
}

}
