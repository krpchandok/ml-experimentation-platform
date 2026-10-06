#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <optional>
#include <poll.h>
#include <string>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "procfs.h"
#include "records.h"
#include "sampler.h"

namespace {

volatile std::sig_atomic_t g_stop_signal = 0;

void handle_stop_signal(int signal_number) { g_stop_signal = signal_number; }

struct Options {
    int pid = 0;
    int interval_ms = 1000;
    std::string out = "-";
    std::string proc_root = "/proc";
};

constexpr const char* kUsage =
    "usage: mlplat-agent --pid <PID> [--interval-ms <ms>] [--out <path|->] [--proc-root <dir>]\n";

std::optional<Options> parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* value = nullptr;
        if (arg == "--pid" && (value = next())) {
            options.pid = std::atoi(value);
        } else if (arg == "--interval-ms" && (value = next())) {
            options.interval_ms = std::atoi(value);
        } else if (arg == "--out" && (value = next())) {
            options.out = value;
        } else if (arg == "--proc-root" && (value = next())) {
            options.proc_root = value;
        } else {
            return std::nullopt;
        }
    }
    if (options.pid <= 0 || options.interval_ms < 10) return std::nullopt;
    return options;
}

double clock_seconds(clockid_t clock) {
    timespec now{};
    clock_gettime(clock, &now);
    return static_cast<double>(now.tv_sec) + static_cast<double>(now.tv_nsec) / 1e9;
}

double monotonic_now() { return clock_seconds(CLOCK_MONOTONIC); }
double wall_now() { return clock_seconds(CLOCK_REALTIME); }

double process_cpu_seconds(uint64_t* max_rss_kb) {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    if (max_rss_kb) {
        mlplat::FileRead status = mlplat::read_file("/proc/self/status");
        auto peak = status.status == mlplat::ReadStatus::Ok ? mlplat::parse_status(status.content).vm_hwm_kb
                                                             : std::nullopt;
        *max_rss_kb = peak.value_or(static_cast<uint64_t>(usage.ru_maxrss));
    }
    auto seconds = [](const timeval& tv) {
        return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1e6;
    };
    return seconds(usage.ru_utime) + seconds(usage.ru_stime);
}

bool delay_accounting_enabled(const std::string& proc_root) {
    mlplat::FileRead read = mlplat::read_file(proc_root + "/sys/kernel/task_delayacct");
    return read.status == mlplat::ReadStatus::Ok && !read.content.empty() && read.content[0] == '1';
}

std::string hostname() {
    char name[256] = {};
    if (gethostname(name, sizeof(name) - 1) != 0) return "";
    return name;
}

int open_pidfd(int pid) {
#ifdef SYS_pidfd_open
    return static_cast<int>(syscall(SYS_pidfd_open, pid, 0));
#else
    errno = ENOSYS;
    return -1;
#endif
}

void install_signal_handlers() {
    struct sigaction action{};
    action.sa_handler = handle_stop_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGHUP, &action, nullptr);
    std::signal(SIGPIPE, SIG_IGN);
}

enum class WaitResult { Deadline, RootExited, Signal };

WaitResult wait_until(double deadline, int pidfd) {
    while (true) {
        if (g_stop_signal) return WaitResult::Signal;
        double remaining = deadline - monotonic_now();
        if (remaining <= 0) return WaitResult::Deadline;
        int timeout_ms = static_cast<int>(remaining * 1000.0) + 1;
        if (pidfd >= 0) {
            pollfd watch{pidfd, POLLIN, 0};
            int ready = poll(&watch, 1, timeout_ms);
            if (ready > 0) return WaitResult::RootExited;
        } else {
            timespec pause{timeout_ms / 1000, static_cast<long>(timeout_ms % 1000) * 1000000L};
            nanosleep(&pause, nullptr);
        }
    }
}

class LineSink {
public:
    explicit LineSink(const std::string& path) {
        file_ = path == "-" ? stdout : std::fopen(path.c_str(), "w");
        owns_ = file_ != stdout;
    }
    ~LineSink() {
        if (file_ && owns_) std::fclose(file_);
    }
    bool ok() const { return file_ != nullptr && !failed_; }
    void write(const std::string& line) {
        if (!file_) return;
        if (std::fwrite(line.data(), 1, line.size(), file_) != line.size() || std::fputc('\n', file_) == EOF ||
            std::fflush(file_) != 0) {
            failed_ = true;
        }
    }

private:
    FILE* file_ = nullptr;
    bool owns_ = false;
    bool failed_ = false;
};

}

int main(int argc, char** argv) {
    auto options = parse_options(argc, argv);
    if (!options) {
        std::cerr << kUsage;
        return 2;
    }
    install_signal_handlers();

    LineSink sink(options->out);
    if (!sink.ok()) {
        std::cerr << "mlplat-agent: cannot open output " << options->out << ": " << std::strerror(errno) << "\n";
        return 2;
    }

    const double start_mono = monotonic_now();
    const double start_cpu = process_cpu_seconds(nullptr);

    mlplat::SamplerConfig config;
    config.proc_root = options->proc_root;
    config.root_pid = options->pid;
    config.clock_ticks_per_second = sysconf(_SC_CLK_TCK);
    config.page_size = sysconf(_SC_PAGESIZE);
    config.delayacct_enabled = delay_accounting_enabled(options->proc_root);

    int pidfd = options->proc_root == "/proc" ? open_pidfd(options->pid) : -1;

    mlplat::HeaderInfo header;
    header.root_pid = options->pid;
    header.interval_ms = options->interval_ms;
    header.clock_ticks_per_second = config.clock_ticks_per_second;
    header.page_size = config.page_size;
    header.online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    header.delayacct_enabled = config.delayacct_enabled;
    header.pidfd_supported = pidfd >= 0;
    header.hostname = hostname();
    header.proc_root = options->proc_root;
    header.t_mono = start_mono;
    header.t_wall = wall_now();
    sink.write(mlplat::format_header(header));

    mlplat::Sampler sampler(config);
    mlplat::EndInfo end;
    auto initial_events = sampler.prime(monotonic_now());
    if (!initial_events) {
        end.reason = "root_not_found";
    } else {
        double now = monotonic_now();
        for (const auto& event : *initial_events) sink.write(mlplat::format_event(event, now));
        end.max_tracked = sampler.tracked_count();
    }

    const double interval = options->interval_ms / 1000.0;
    double deadline = monotonic_now() + interval;
    double sample_us_total = 0;

    while (end.reason.empty()) {
        WaitResult waited = wait_until(deadline, pidfd);

        double sample_started = monotonic_now();
        double wall = wall_now();
        mlplat::Sample sample = sampler.sample(sample_started);
        end.max_tracked = std::max<uint64_t>({end.max_tracked, sampler.tracked_count(), sample.processes.size()});

        mlplat::AgentUsage usage;
        usage.cpu_seconds = process_cpu_seconds(&usage.max_rss_kb) - start_cpu;
        usage.sample_us = (monotonic_now() - sample_started) * 1e6;

        for (const auto& event : sample.events) sink.write(mlplat::format_event(event, sample_started));
        sink.write(mlplat::format_sample(sample, ++end.samples, sample_started, wall, usage));

        sample_us_total += usage.sample_us;
        end.sample_us_max = std::max(end.sample_us_max, usage.sample_us);

        if (!sink.ok()) end.reason = "write_error";
        else if (!sample.root_alive) end.reason = "root_exited";
        else if (waited == WaitResult::Signal) end.reason = "signal";
        else if (waited == WaitResult::RootExited) end.reason = "root_exited";

        deadline += interval;
        double after = monotonic_now();
        if (deadline < after) deadline = after + interval;
    }

    if (pidfd >= 0) close(pidfd);
    end.t_mono = monotonic_now();
    end.t_wall = wall_now();
    end.agent_cpu_seconds = process_cpu_seconds(&end.max_rss_kb) - start_cpu;
    end.agent_wall_seconds = end.t_mono - start_mono;
    end.sample_us_mean = end.samples ? sample_us_total / static_cast<double>(end.samples) : 0;
    sink.write(mlplat::format_end(end));
    return end.reason == "root_not_found" ? 3 : 0;
}
