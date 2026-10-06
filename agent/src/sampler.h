#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "procfs.h"

namespace mlplat {

struct SamplerConfig {
    std::string proc_root = "/proc";
    int root_pid = 0;
    long clock_ticks_per_second = 100;
    long page_size = 4096;
    bool delayacct_enabled = false;
};

struct ProcessCounters {
    uint64_t utime = 0;
    uint64_t stime = 0;
    uint64_t minflt = 0;
    uint64_t majflt = 0;
    std::optional<uint64_t> voluntary_switches;
    std::optional<uint64_t> nonvoluntary_switches;
    std::optional<uint64_t> read_bytes;
    std::optional<uint64_t> write_bytes;
    std::optional<uint64_t> blkio_ticks;
};

struct ProcessSample {
    int pid = 0;
    int ppid = 0;
    std::string comm;
    char state = '?';
    int64_t threads = 0;
    double cpu_pct = 0;
    double user_pct = 0;
    double system_pct = 0;
    double cpu_seconds = 0;
    uint64_t rss_kb = 0;
    std::optional<uint64_t> read_bytes;
    std::optional<uint64_t> write_bytes;
    std::optional<double> minflt_per_s;
    std::optional<double> majflt_per_s;
    std::optional<double> voluntary_switches_per_s;
    std::optional<double> nonvoluntary_switches_per_s;
    std::optional<double> read_bytes_per_s;
    std::optional<double> write_bytes_per_s;
    std::optional<double> blkio_wait_pct;
    std::vector<std::string> unavailable;
};

enum class EventKind { Started, Exec, Exited };

struct ProcessEvent {
    EventKind kind = EventKind::Started;
    int pid = 0;
    int ppid = 0;
    uint64_t start_ticks = 0;
    std::string comm;
    std::string cmdline;
    double cpu_seconds = 0;
};

struct SystemSample {
    int cpu_count = 0;
    std::optional<double> cpu_busy_pct;
    std::optional<double> cpu_iowait_pct;
    std::optional<double> cores_busy;
    std::optional<double> context_switches_per_s;
    std::optional<uint64_t> procs_running;
    std::optional<uint64_t> procs_blocked;
    std::optional<uint64_t> mem_total_kb;
    std::optional<uint64_t> mem_available_kb;
    std::optional<uint64_t> swap_used_kb;
};

struct Sample {
    double dt = 0;
    bool root_alive = true;
    std::vector<ProcessEvent> events;
    std::vector<ProcessSample> processes;
    SystemSample system;
};

class Sampler {
public:
    explicit Sampler(SamplerConfig config);

    std::optional<std::vector<ProcessEvent>> prime(double now);
    Sample sample(double now);
    std::size_t tracked_count() const { return tracked_.size(); }

private:
    struct Tracked {
        int ppid = 0;
        uint64_t start_ticks = 0;
        std::string comm;
        ProcessCounters counters;
    };

    struct Observation {
        StatInfo stat;
        std::optional<StatusInfo> status;
        std::optional<IoInfo> io;
    };

    std::string path(int pid, const char* file) const;
    std::optional<StatInfo> read_stat(int pid) const;
    std::optional<Observation> observe(int pid) const;
    std::string read_cmdline(int pid, const std::string& comm) const;
    ProcessCounters counters_from(const Observation& observation) const;
    void discover(const std::vector<int>& pids, bool newborn, std::vector<ProcessEvent>& events);
    ProcessSample measure(int pid, const Observation& observation, const ProcessCounters& previous,
                          double dt) const;
    SystemSample measure_system(double dt);

    SamplerConfig config_;
    std::unordered_map<int, Tracked> tracked_;
    std::unordered_set<int> outsiders_;
    std::optional<SystemStat> previous_system_;
    double previous_time_ = 0;
};

}
