#include "nvml.h"

#include <dlfcn.h>

#include <cstring>

namespace mlplat {

namespace {

using Return = int;
using Device = void*;

constexpr Return kSuccess = 0;
constexpr Return kInsufficientSize = 7;
constexpr Return kNotFound = 6;
constexpr int kTemperatureGpu = 0;
constexpr int kGpuUtilizationSamples = 1;
constexpr int kMemoryUtilizationSamples = 2;
constexpr unsigned long long kValueNotAvailable = ~0ULL;
constexpr double kBytesPerMib = 1024.0 * 1024.0;
constexpr unsigned kMaxProcesses = 256;

struct Utilization {
    unsigned gpu;
    unsigned memory;
};

struct Memory {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

struct ProcessInfoV1 {
    unsigned pid;
    unsigned long long used_gpu_memory;
};

struct ProcessInfoV2 {
    unsigned pid;
    unsigned long long used_gpu_memory;
    unsigned gpu_instance_id;
    unsigned compute_instance_id;
};

union Value {
    double d;
    unsigned ui;
    unsigned long ul;
    unsigned long long ull;
    long long sll;
};

struct SampleValue {
    unsigned long long timestamp;
    Value value;
};

double value_as_double(int type, const Value& value) {
    switch (type) {
        case 0: return value.d;
        case 1: return static_cast<double>(value.ui);
        case 2: return static_cast<double>(value.ul);
        case 3: return static_cast<double>(value.ull);
        case 4: return static_cast<double>(value.sll);
        default: return -1;
    }
}

template <typename Function>
Function symbol(void* handle, const char* name) {
    return reinterpret_cast<Function>(dlsym(handle, name));
}

}

struct NvmlSampler::Api {
    Return (*init)() = nullptr;
    Return (*shutdown)() = nullptr;
    const char* (*error_string)(Return) = nullptr;
    Return (*device_count)(unsigned*) = nullptr;
    Return (*device_by_index)(unsigned, Device*) = nullptr;
    Return (*device_name)(Device, char*, unsigned) = nullptr;
    Return (*device_uuid)(Device, char*, unsigned) = nullptr;
    Return (*driver_version)(char*, unsigned) = nullptr;
    Return (*utilization)(Device, Utilization*) = nullptr;
    Return (*memory)(Device, Memory*) = nullptr;
    Return (*power)(Device, unsigned*) = nullptr;
    Return (*temperature)(Device, int, unsigned*) = nullptr;
    Return (*samples)(Device, int, unsigned long long, int*, unsigned*, SampleValue*) = nullptr;
    Return (*processes_v2_layout)(Device, unsigned*, ProcessInfoV2*) = nullptr;
    Return (*processes_v1_layout)(Device, unsigned*, ProcessInfoV1*) = nullptr;

    std::string describe(Return code) const {
        const char* text = error_string ? error_string(code) : nullptr;
        return text ? text : "NVML error " + std::to_string(code);
    }
};

std::unique_ptr<NvmlSampler> NvmlSampler::open(const std::string& library, std::string& failure) {
    void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* reason = dlerror();
        failure = std::string("cannot load ") + library + ": " + (reason ? reason : "unknown error");
        return nullptr;
    }

    auto api = std::make_unique<Api>();
    api->init = symbol<decltype(api->init)>(handle, "nvmlInit_v2");
    if (!api->init) api->init = symbol<decltype(api->init)>(handle, "nvmlInit");
    api->shutdown = symbol<decltype(api->shutdown)>(handle, "nvmlShutdown");
    api->error_string = symbol<decltype(api->error_string)>(handle, "nvmlErrorString");
    api->device_count = symbol<decltype(api->device_count)>(handle, "nvmlDeviceGetCount_v2");
    if (!api->device_count) api->device_count = symbol<decltype(api->device_count)>(handle, "nvmlDeviceGetCount");
    api->device_by_index = symbol<decltype(api->device_by_index)>(handle, "nvmlDeviceGetHandleByIndex_v2");
    if (!api->device_by_index) {
        api->device_by_index = symbol<decltype(api->device_by_index)>(handle, "nvmlDeviceGetHandleByIndex");
    }
    api->device_name = symbol<decltype(api->device_name)>(handle, "nvmlDeviceGetName");
    api->device_uuid = symbol<decltype(api->device_uuid)>(handle, "nvmlDeviceGetUUID");
    api->driver_version = symbol<decltype(api->driver_version)>(handle, "nvmlSystemGetDriverVersion");
    api->utilization = symbol<decltype(api->utilization)>(handle, "nvmlDeviceGetUtilizationRates");
    api->memory = symbol<decltype(api->memory)>(handle, "nvmlDeviceGetMemoryInfo");
    api->power = symbol<decltype(api->power)>(handle, "nvmlDeviceGetPowerUsage");
    api->temperature = symbol<decltype(api->temperature)>(handle, "nvmlDeviceGetTemperature");
    api->samples = symbol<decltype(api->samples)>(handle, "nvmlDeviceGetSamples");
    api->processes_v2_layout = symbol<decltype(api->processes_v2_layout)>(handle, "nvmlDeviceGetComputeRunningProcesses_v3");
    if (!api->processes_v2_layout) {
        api->processes_v2_layout =
            symbol<decltype(api->processes_v2_layout)>(handle, "nvmlDeviceGetComputeRunningProcesses_v2");
    }
    if (!api->processes_v2_layout) {
        api->processes_v1_layout =
            symbol<decltype(api->processes_v1_layout)>(handle, "nvmlDeviceGetComputeRunningProcesses");
    }

    if (!api->init || !api->shutdown || !api->device_count || !api->device_by_index) {
        failure = "required NVML symbols are missing from " + library;
        dlclose(handle);
        return nullptr;
    }
    Return status = api->init();
    if (status != kSuccess) {
        failure = "nvmlInit failed: " + api->describe(status);
        dlclose(handle);
        return nullptr;
    }

    std::unique_ptr<NvmlSampler> sampler(new NvmlSampler());
    sampler->handle_ = handle;
    sampler->api_ = std::move(api);

    char text[256] = {};
    if (sampler->api_->driver_version && sampler->api_->driver_version(text, sizeof(text)) == kSuccess) {
        sampler->driver_version_ = text;
    }
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(sampler->api_->init), &info) && info.dli_fname) {
        sampler->library_path_ = info.dli_fname;
    }

    unsigned count = 0;
    status = sampler->api_->device_count(&count);
    if (status != kSuccess) {
        failure = "nvmlDeviceGetCount failed: " + sampler->api_->describe(status);
        return nullptr;
    }
    if (count == 0) {
        failure = "NVML reports no GPU devices";
        return nullptr;
    }
    for (unsigned index = 0; index < count; ++index) {
        Device device = nullptr;
        if (sampler->api_->device_by_index(index, &device) != kSuccess) continue;
        GpuDeviceInfo device_info;
        device_info.index = index;
        std::memset(text, 0, sizeof(text));
        if (sampler->api_->device_name && sampler->api_->device_name(device, text, sizeof(text)) == kSuccess) {
            device_info.name = text;
        }
        std::memset(text, 0, sizeof(text));
        if (sampler->api_->device_uuid && sampler->api_->device_uuid(device, text, sizeof(text)) == kSuccess) {
            device_info.uuid = text;
        }
        Memory memory{};
        if (sampler->api_->memory && sampler->api_->memory(device, &memory) == kSuccess) {
            device_info.memory_total_mb = static_cast<double>(memory.total) / kBytesPerMib;
        }
        sampler->device_handles_.push_back(device);
        sampler->devices_.push_back(device_info);
    }
    if (sampler->devices_.empty()) {
        failure = "no NVML device handle could be opened";
        return nullptr;
    }
    sampler->last_util_timestamp_.assign(sampler->devices_.size(), 0);
    sampler->last_mem_timestamp_.assign(sampler->devices_.size(), 0);
    return sampler;
}

NvmlSampler::~NvmlSampler() {
    if (api_ && api_->shutdown) api_->shutdown();
    if (handle_) dlclose(handle_);
}

std::optional<double> NvmlSampler::average_samples(std::size_t device, int sampling_type, uint64_t& last_seen) {
    if (!api_->samples) return std::nullopt;
    int value_type = 0;
    unsigned count = 0;
    Return status = api_->samples(device_handles_[device], sampling_type, last_seen, &value_type, &count, nullptr);
    if (status != kSuccess && status != kInsufficientSize) return std::nullopt;
    if (count == 0) return std::nullopt;
    std::vector<SampleValue> buffer(count);
    status = api_->samples(device_handles_[device], sampling_type, last_seen, &value_type, &count, buffer.data());
    if (status != kSuccess || count == 0) return std::nullopt;

    double total = 0;
    unsigned used = 0;
    uint64_t newest = last_seen;
    for (unsigned i = 0; i < count; ++i) {
        if (buffer[i].timestamp <= last_seen) continue;
        double value = value_as_double(value_type, buffer[i].value);
        if (value < 0) continue;
        total += value;
        ++used;
        if (buffer[i].timestamp > newest) newest = buffer[i].timestamp;
    }
    if (used == 0) return std::nullopt;
    bool first_read = last_seen == 0;
    last_seen = newest;
    if (first_read) return std::nullopt;
    return total / used;
}

void NvmlSampler::sample_processes(std::size_t device, const std::unordered_set<int>& tracked_pids,
                                   GpuSample& out) {
    unsigned count = kMaxProcesses;
    Return status = kNotFound;
    std::vector<std::pair<unsigned, unsigned long long>> found;
    if (api_->processes_v2_layout) {
        std::vector<ProcessInfoV2> buffer(kMaxProcesses);
        status = api_->processes_v2_layout(device_handles_[device], &count, buffer.data());
        if (status == kSuccess) {
            for (unsigned i = 0; i < count; ++i) found.emplace_back(buffer[i].pid, buffer[i].used_gpu_memory);
        }
    } else if (api_->processes_v1_layout) {
        std::vector<ProcessInfoV1> buffer(kMaxProcesses);
        status = api_->processes_v1_layout(device_handles_[device], &count, buffer.data());
        if (status == kSuccess) {
            for (unsigned i = 0; i < count; ++i) found.emplace_back(buffer[i].pid, buffer[i].used_gpu_memory);
        }
    }
    if (status != kSuccess) return;
    out.process_info = out.process_info || !found.empty();
    for (const auto& [pid, used] : found) {
        if (!tracked_pids.count(static_cast<int>(pid))) continue;
        GpuProcessSample process;
        process.pid = static_cast<int>(pid);
        process.device = devices_[device].index;
        if (used != kValueNotAvailable) process.mem_used_mb = static_cast<double>(used) / kBytesPerMib;
        out.processes.push_back(process);
    }
}

GpuSample NvmlSampler::sample(const std::unordered_set<int>& tracked_pids) {
    GpuSample out;
    for (std::size_t device = 0; device < device_handles_.size(); ++device) {
        GpuDeviceSample sample;
        sample.index = devices_[device].index;
        Device handle = device_handles_[device];

        auto averaged_util = average_samples(device, kGpuUtilizationSamples, last_util_timestamp_[device]);
        auto averaged_mem = average_samples(device, kMemoryUtilizationSamples, last_mem_timestamp_[device]);
        Utilization rates{};
        bool have_rates = api_->utilization && api_->utilization(handle, &rates) == kSuccess;
        if (averaged_util) {
            sample.util_pct = averaged_util;
            sample.util_source = "samples";
        } else if (have_rates) {
            sample.util_pct = rates.gpu;
            sample.util_source = "rates";
        } else {
            sample.unavailable.push_back("util");
        }
        if (averaged_mem) {
            sample.mem_util_pct = averaged_mem;
        } else if (have_rates) {
            sample.mem_util_pct = rates.memory;
        }

        Memory memory{};
        if (api_->memory && api_->memory(handle, &memory) == kSuccess) {
            sample.mem_used_mb = static_cast<double>(memory.used) / kBytesPerMib;
            sample.mem_total_mb = static_cast<double>(memory.total) / kBytesPerMib;
        } else {
            sample.unavailable.push_back("memory");
        }
        unsigned milliwatts = 0;
        if (api_->power && api_->power(handle, &milliwatts) == kSuccess) {
            sample.power_w = milliwatts / 1000.0;
        } else {
            sample.unavailable.push_back("power");
        }
        unsigned celsius = 0;
        if (api_->temperature && api_->temperature(handle, kTemperatureGpu, &celsius) == kSuccess) {
            sample.temp_c = celsius;
        } else {
            sample.unavailable.push_back("temperature");
        }
        out.devices.push_back(std::move(sample));
        sample_processes(device, tracked_pids, out);
    }
    return out;
}

}
