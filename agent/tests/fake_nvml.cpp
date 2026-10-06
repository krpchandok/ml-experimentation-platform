#include <cstdlib>
#include <cstring>

namespace {

struct Utilization {
    unsigned gpu;
    unsigned memory;
};

struct Memory {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

struct ProcessInfo {
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

constexpr int kSuccess = 0;
constexpr int kUninitialized = 1;
constexpr int kInvalidArgument = 2;
constexpr int kNotSupported = 3;
constexpr int kInsufficientSize = 7;
constexpr unsigned long long kMib = 1024ULL * 1024ULL;
constexpr unsigned kSamplesPerCall = 4;

bool g_initialized = false;
unsigned long long g_sample_clock = 1000;

long env_number(const char* name, long fallback) {
    const char* value = std::getenv(name);
    return value ? std::strtol(value, nullptr, 10) : fallback;
}

bool env_flag(const char* name) { return env_number(name, 0) != 0; }

unsigned device_count() { return static_cast<unsigned>(env_number("FAKE_NVML_DEVICES", 1)); }

unsigned device_index(void* device) { return static_cast<unsigned>(reinterpret_cast<unsigned long>(device) - 1); }

unsigned utilization_for(unsigned index) {
    return static_cast<unsigned>(env_number("FAKE_NVML_UTIL", 40) + 10 * static_cast<long>(index));
}

}

extern "C" {

int nvmlInit_v2() {
    if (env_flag("FAKE_NVML_INIT_FAIL")) return kNotSupported;
    g_initialized = true;
    return kSuccess;
}

int nvmlShutdown() {
    g_initialized = false;
    return kSuccess;
}

const char* nvmlErrorString(int code) { return code == kNotSupported ? "Not Supported" : "Fake Error"; }

int nvmlSystemGetDriverVersion(char* text, unsigned length) {
    std::strncpy(text, "999.99-fake", length);
    return kSuccess;
}

int nvmlDeviceGetCount_v2(unsigned* count) {
    if (!g_initialized) return kUninitialized;
    *count = device_count();
    return kSuccess;
}

int nvmlDeviceGetHandleByIndex_v2(unsigned index, void** device) {
    if (index >= device_count()) return kInvalidArgument;
    *device = reinterpret_cast<void*>(static_cast<unsigned long>(index) + 1);
    return kSuccess;
}

int nvmlDeviceGetName(void* device, char* text, unsigned length) {
    std::strncpy(text, device_index(device) == 0 ? "Fake GPU 0" : "Fake GPU N", length);
    return kSuccess;
}

int nvmlDeviceGetUUID(void* device, char* text, unsigned length) {
    std::strncpy(text, device_index(device) == 0 ? "GPU-fake-0" : "GPU-fake-n", length);
    return kSuccess;
}

int nvmlDeviceGetUtilizationRates(void* device, Utilization* rates) {
    rates->gpu = utilization_for(device_index(device));
    rates->memory = 20;
    return kSuccess;
}

int nvmlDeviceGetMemoryInfo(void* device, Memory* memory) {
    memory->total = 8192 * kMib;
    memory->used = static_cast<unsigned long long>(1024 + 512 * device_index(device)) * kMib;
    memory->free = memory->total - memory->used;
    return kSuccess;
}

int nvmlDeviceGetPowerUsage(void*, unsigned* milliwatts) {
    if (env_flag("FAKE_NVML_NO_POWER")) return kNotSupported;
    *milliwatts = 115500;
    return kSuccess;
}

int nvmlDeviceGetTemperature(void*, int, unsigned* celsius) {
    *celsius = 61;
    return kSuccess;
}

int nvmlDeviceGetSamples(void* device, int type, unsigned long long last_seen, int* value_type, unsigned* count,
                         SampleValue* samples) {
    if (env_flag("FAKE_NVML_NO_SAMPLES")) return kNotSupported;
    *value_type = 1;
    if (samples == nullptr) {
        *count = kSamplesPerCall;
        return kSuccess;
    }
    if (*count < kSamplesPerCall) return kInsufficientSize;
    unsigned base = type == 1 ? utilization_for(device_index(device)) : 20;
    unsigned long long start = last_seen > g_sample_clock ? last_seen : g_sample_clock;
    for (unsigned i = 0; i < kSamplesPerCall; ++i) {
        samples[i].timestamp = start + i + 1;
        samples[i].value.ui = base - 3 + 2 * i;
    }
    g_sample_clock = start + kSamplesPerCall;
    *count = kSamplesPerCall;
    return kSuccess;
}

int nvmlDeviceGetComputeRunningProcesses_v3(void* device, unsigned* count, ProcessInfo* infos) {
    long pid = env_number("FAKE_NVML_PID", 0);
    if (pid <= 0 || device_index(device) != 0) {
        *count = 0;
        return kSuccess;
    }
    if (*count < 2) {
        *count = 2;
        return kInsufficientSize;
    }
    infos[0] = ProcessInfo{static_cast<unsigned>(pid), 512 * kMib, 0, 0};
    infos[1] = ProcessInfo{999999, ~0ULL, 0, 0};
    *count = 2;
    return kSuccess;
}

}
