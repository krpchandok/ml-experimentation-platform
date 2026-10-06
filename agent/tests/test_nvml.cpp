#include <algorithm>
#include <cstdlib>
#include <unistd.h>

#include "check.h"
#include "nvml.h"
#include "records.h"

using namespace mlplat;

namespace {

class FakeEnvironment {
public:
    explicit FakeEnvironment(std::initializer_list<std::pair<const char*, std::string>> values) {
        for (const auto& [name, value] : values) {
            names_.push_back(name);
            setenv(name, value.c_str(), 1);
        }
    }
    ~FakeEnvironment() {
        for (const char* name : names_) unsetenv(name);
    }

private:
    std::vector<const char*> names_;
};

std::unique_ptr<NvmlSampler> open_fake(std::string& failure) { return NvmlSampler::open(MLPLAT_FAKE_NVML, failure); }

bool contains(const std::vector<std::string>& values, const std::string& wanted) {
    return std::find(values.begin(), values.end(), wanted) != values.end();
}

}

TEST_CASE("missing NVML library is reported, not fatal") {
    std::string failure;
    auto sampler = NvmlSampler::open("/definitely/not/libnvidia-ml.so.1", failure);
    CHECK(sampler == nullptr);
    CHECK(failure.find("cannot load /definitely/not/libnvidia-ml.so.1") == 0);
}

TEST_CASE("failed nvmlInit is reported with the NVML error string") {
    FakeEnvironment environment({{"FAKE_NVML_INIT_FAIL", "1"}});
    std::string failure;
    CHECK(open_fake(failure) == nullptr);
    CHECK_EQ(failure, std::string("nvmlInit failed: Not Supported"));
}

TEST_CASE("devices are discovered with names and memory") {
    FakeEnvironment environment({{"FAKE_NVML_DEVICES", "2"}});
    std::string failure;
    auto sampler = open_fake(failure);
    REQUIRE(sampler != nullptr);
    REQUIRE(sampler->devices().size() == 2);
    CHECK_EQ(sampler->devices()[0].name, std::string("Fake GPU 0"));
    CHECK_EQ(sampler->devices()[0].uuid, std::string("GPU-fake-0"));
    CHECK_NEAR(sampler->devices()[0].memory_total_mb.value_or(0), 8192.0, 1e-9);
    CHECK_EQ(sampler->driver_version(), std::string("999.99-fake"));
    CHECK(sampler->library_path().find("libfake-nvidia-ml") != std::string::npos);
}

TEST_CASE("utilization uses rates first, then the averaged sample buffer") {
    FakeEnvironment environment({{"FAKE_NVML_UTIL", "60"}});
    std::string failure;
    auto sampler = open_fake(failure);
    REQUIRE(sampler != nullptr);
    GpuSample first = sampler->sample({});
    REQUIRE(first.devices.size() == 1);
    CHECK_EQ(first.devices[0].util_source, std::string("rates"));
    CHECK_NEAR(first.devices[0].util_pct.value_or(-1), 60.0, 1e-9);
    GpuSample second = sampler->sample({});
    CHECK_EQ(second.devices[0].util_source, std::string("samples"));
    CHECK_NEAR(second.devices[0].util_pct.value_or(-1), 60.0, 1e-9);
    CHECK_NEAR(second.devices[0].mem_used_mb.value_or(-1), 1024.0, 1e-9);
    CHECK_NEAR(second.devices[0].power_w.value_or(-1), 115.5, 1e-9);
    CHECK_NEAR(second.devices[0].temp_c.value_or(-1), 61.0, 1e-9);
    CHECK(second.devices[0].unavailable.empty());
}

TEST_CASE("unsupported fields are listed as unavailable") {
    FakeEnvironment environment({{"FAKE_NVML_NO_POWER", "1"}, {"FAKE_NVML_NO_SAMPLES", "1"}});
    std::string failure;
    auto sampler = open_fake(failure);
    REQUIRE(sampler != nullptr);
    sampler->sample({});
    GpuSample sample = sampler->sample({});
    CHECK_EQ(sample.devices[0].util_source, std::string("rates"));
    CHECK(!sample.devices[0].power_w.has_value());
    CHECK(contains(sample.devices[0].unavailable, "power"));
}

TEST_CASE("per-process GPU memory is reported only for tracked processes") {
    int self = static_cast<int>(getpid());
    FakeEnvironment environment({{"FAKE_NVML_PID", std::to_string(self)}, {"FAKE_NVML_DEVICES", "2"}});
    std::string failure;
    auto sampler = open_fake(failure);
    REQUIRE(sampler != nullptr);
    GpuSample sample = sampler->sample({self, 12345});
    CHECK(sample.process_info);
    REQUIRE(sample.processes.size() == 1);
    CHECK_EQ(sample.processes[0].pid, self);
    CHECK_EQ(sample.processes[0].device, 0u);
    CHECK_NEAR(sample.processes[0].mem_used_mb.value_or(-1), 512.0, 1e-9);
    CHECK(sampler->sample({12345}).processes.empty());
}

TEST_CASE("gpu records are written into header and samples") {
    GpuHeader header;
    header.reason = "cannot load libnvidia-ml.so.1";
    HeaderInfo info;
    info.gpu = header;
    CHECK(format_header(info).find("\"gpu\":{\"available\":false,\"reason\":\"cannot load libnvidia-ml.so.1\"}") !=
          std::string::npos);
    Sample sample;
    CHECK(format_sample(sample, 1, 0, 0, AgentUsage{}).find("\"gpu\":\"unavailable\"") != std::string::npos);

    GpuSample gpu;
    GpuDeviceSample device;
    device.util_pct = 87.5;
    device.util_source = "samples";
    device.unavailable = {"power"};
    gpu.devices = {device};
    gpu.processes = {GpuProcessSample{42, 0, 300.0}};
    std::string line = format_sample(sample, 1, 0, 0, AgentUsage{}, gpu);
    CHECK(line.find("\"util_pct\":87.5,\"util_source\":\"samples\"") != std::string::npos);
    CHECK(line.find("\"unavailable\":[\"power\"]") != std::string::npos);
    CHECK(line.find("\"procs\":[{\"pid\":42,\"device\":0,\"mem_used_mb\":300}]") != std::string::npos);
}
