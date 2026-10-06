#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace mlplat {

struct GpuDeviceInfo {
    unsigned index = 0;
    std::string name;
    std::string uuid;
    std::optional<double> memory_total_mb;
};

struct GpuDeviceSample {
    unsigned index = 0;
    std::optional<double> util_pct;
    std::optional<double> mem_util_pct;
    std::optional<double> mem_used_mb;
    std::optional<double> mem_total_mb;
    std::optional<double> power_w;
    std::optional<double> temp_c;
    std::string util_source;
    std::vector<std::string> unavailable;
};

struct GpuProcessSample {
    int pid = 0;
    unsigned device = 0;
    std::optional<double> mem_used_mb;
};

struct GpuSample {
    std::vector<GpuDeviceSample> devices;
    std::vector<GpuProcessSample> processes;
    bool process_info = false;
};

class NvmlSampler {
public:
    static std::unique_ptr<NvmlSampler> open(const std::string& library, std::string& failure);
    ~NvmlSampler();
    NvmlSampler(const NvmlSampler&) = delete;
    NvmlSampler& operator=(const NvmlSampler&) = delete;

    const std::vector<GpuDeviceInfo>& devices() const { return devices_; }
    const std::string& driver_version() const { return driver_version_; }
    const std::string& library_path() const { return library_path_; }
    GpuSample sample(const std::unordered_set<int>& tracked_pids);

    struct Api;

private:
    NvmlSampler() = default;

    std::optional<double> average_samples(std::size_t device, int sampling_type, uint64_t& last_seen);
    void sample_processes(std::size_t device, const std::unordered_set<int>& tracked_pids, GpuSample& out);

    void* handle_ = nullptr;
    std::unique_ptr<Api> api_;
    std::vector<void*> device_handles_;
    std::vector<GpuDeviceInfo> devices_;
    std::vector<uint64_t> last_util_timestamp_;
    std::vector<uint64_t> last_mem_timestamp_;
    std::string driver_version_;
    std::string library_path_;
};

}
