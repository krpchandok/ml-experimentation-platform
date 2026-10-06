#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nvml.h"
#include "sampler.h"

namespace mlplat {

struct GpuHeader {
    bool available = false;
    std::string reason;
    std::string driver_version;
    std::string library;
    std::vector<GpuDeviceInfo> devices;
    double init_ms = 0;
};

struct HeaderInfo {
    int root_pid = 0;
    int interval_ms = 0;
    long clock_ticks_per_second = 0;
    long page_size = 0;
    long online_cpus = 0;
    bool delayacct_enabled = false;
    bool pidfd_supported = false;
    std::string hostname;
    std::string proc_root;
    double t_mono = 0;
    double t_wall = 0;
    GpuHeader gpu;
};

struct AgentUsage {
    double cpu_seconds = 0;
    uint64_t max_rss_kb = 0;
    double sample_us = 0;
};

struct EndInfo {
    std::string reason;
    uint64_t samples = 0;
    double t_mono = 0;
    double t_wall = 0;
    double agent_cpu_seconds = 0;
    double startup_cpu_seconds = 0;
    double agent_wall_seconds = 0;
    double sample_us_mean = 0;
    double sample_us_max = 0;
    uint64_t max_rss_kb = 0;
    uint64_t max_tracked = 0;
};

std::string format_header(const HeaderInfo& header);
std::string format_event(const ProcessEvent& event, double t_mono);
std::string format_sample(const Sample& sample, uint64_t seq, double t_mono, double t_wall,
                          const AgentUsage& agent, const std::optional<GpuSample>& gpu = std::nullopt);
std::string format_end(const EndInfo& end);

}
