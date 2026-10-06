#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mlplat {

enum class ReadStatus { Ok, NotFound, PermissionDenied, Error };

struct FileRead {
    ReadStatus status = ReadStatus::Error;
    std::string content;
};

FileRead read_file(const std::string& path);

struct StatInfo {
    int pid = 0;
    std::string comm;
    char state = '?';
    int ppid = 0;
    uint64_t minflt = 0;
    uint64_t majflt = 0;
    uint64_t utime = 0;
    uint64_t stime = 0;
    int64_t cutime = 0;
    int64_t cstime = 0;
    int64_t num_threads = 0;
    uint64_t starttime = 0;
    int64_t rss_pages = 0;
    std::optional<uint64_t> delayacct_blkio_ticks;
};

std::optional<StatInfo> parse_stat(std::string_view content);

struct StatusInfo {
    std::optional<uint64_t> vm_rss_kb;
    std::optional<uint64_t> vm_hwm_kb;
    std::optional<uint64_t> voluntary_ctxt_switches;
    std::optional<uint64_t> nonvoluntary_ctxt_switches;
};

StatusInfo parse_status(std::string_view content);

struct IoInfo {
    std::optional<uint64_t> rchar;
    std::optional<uint64_t> wchar;
    std::optional<uint64_t> read_bytes;
    std::optional<uint64_t> write_bytes;
    std::optional<uint64_t> cancelled_write_bytes;
};

IoInfo parse_io(std::string_view content);

std::string parse_cmdline(std::string_view content, std::size_t max_length = 2048);

struct CpuTimes {
    uint64_t user = 0;
    uint64_t nice = 0;
    uint64_t system = 0;
    uint64_t idle = 0;
    uint64_t iowait = 0;
    uint64_t irq = 0;
    uint64_t softirq = 0;
    uint64_t steal = 0;

    uint64_t total() const { return user + nice + system + idle + iowait + irq + softirq + steal; }
    uint64_t busy() const { return total() - idle - iowait; }
};

struct SystemStat {
    CpuTimes cpu;
    int cpu_count = 0;
    std::optional<uint64_t> ctxt;
    std::optional<uint64_t> procs_running;
    std::optional<uint64_t> procs_blocked;
};

std::optional<SystemStat> parse_system_stat(std::string_view content);

struct MemInfo {
    std::optional<uint64_t> mem_total_kb;
    std::optional<uint64_t> mem_free_kb;
    std::optional<uint64_t> mem_available_kb;
    std::optional<uint64_t> cached_kb;
    std::optional<uint64_t> swap_total_kb;
    std::optional<uint64_t> swap_free_kb;
};

MemInfo parse_meminfo(std::string_view content);

std::optional<uint64_t> find_field(std::string_view content, std::string_view key);

std::vector<int> list_pids(const std::string& proc_root);

}
