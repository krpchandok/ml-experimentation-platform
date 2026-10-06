#include "sampler.h"

#include <algorithm>

#include "process_tree.h"

namespace mlplat {

namespace {

bool is_dead_state(char state) { return state == 'Z' || state == 'X' || state == 'x'; }

uint64_t saturating_delta(uint64_t current, uint64_t previous) {
    return current >= previous ? current - previous : 0;
}

std::optional<double> rate(const std::optional<uint64_t>& current,
                           const std::optional<uint64_t>& previous, double dt) {
    if (!current || !previous || dt <= 0) return std::nullopt;
    return static_cast<double>(saturating_delta(*current, *previous)) / dt;
}

ProcessCounters zero_counters_like(const ProcessCounters& shape) {
    ProcessCounters zero;
    auto zero_if_present = [](const std::optional<uint64_t>& value) {
        return value ? std::optional<uint64_t>(0) : std::nullopt;
    };
    zero.voluntary_switches = zero_if_present(shape.voluntary_switches);
    zero.nonvoluntary_switches = zero_if_present(shape.nonvoluntary_switches);
    zero.read_bytes = zero_if_present(shape.read_bytes);
    zero.write_bytes = zero_if_present(shape.write_bytes);
    zero.blkio_ticks = zero_if_present(shape.blkio_ticks);
    return zero;
}

}

Sampler::Sampler(SamplerConfig config) : config_(std::move(config)) {}

std::string Sampler::path(int pid, const char* file) const {
    return config_.proc_root + "/" + std::to_string(pid) + "/" + file;
}

std::optional<StatInfo> Sampler::read_stat(int pid) const {
    FileRead read = read_file(path(pid, "stat"));
    if (read.status != ReadStatus::Ok) return std::nullopt;
    return parse_stat(read.content);
}

std::optional<Sampler::Observation> Sampler::observe(int pid) const {
    auto stat = read_stat(pid);
    if (!stat || is_dead_state(stat->state)) return std::nullopt;
    Observation observation{*stat, std::nullopt, std::nullopt};

    FileRead status = read_file(path(pid, "status"));
    if (status.status == ReadStatus::Ok) observation.status = parse_status(status.content);

    FileRead io = read_file(path(pid, "io"));
    if (io.status == ReadStatus::Ok) observation.io = parse_io(io.content);
    return observation;
}

std::string Sampler::read_cmdline(int pid, const std::string& comm) const {
    FileRead read = read_file(path(pid, "cmdline"));
    std::string cmdline = read.status == ReadStatus::Ok ? parse_cmdline(read.content) : "";
    return cmdline.empty() ? "[" + comm + "]" : cmdline;
}

ProcessCounters Sampler::counters_from(const Observation& observation) const {
    ProcessCounters counters;
    counters.utime = observation.stat.utime;
    counters.stime = observation.stat.stime;
    counters.minflt = observation.stat.minflt;
    counters.majflt = observation.stat.majflt;
    if (observation.status) {
        counters.voluntary_switches = observation.status->voluntary_ctxt_switches;
        counters.nonvoluntary_switches = observation.status->nonvoluntary_ctxt_switches;
    }
    if (observation.io) {
        counters.read_bytes = observation.io->read_bytes;
        counters.write_bytes = observation.io->write_bytes;
    }
    if (config_.delayacct_enabled) counters.blkio_ticks = observation.stat.delayacct_blkio_ticks;
    return counters;
}

std::optional<std::vector<ProcessEvent>> Sampler::prime(double now) {
    auto root = observe(config_.root_pid);
    if (!root) return std::nullopt;

    std::vector<ProcessEvent> events;
    tracked_[config_.root_pid] = Tracked{root->stat.ppid, root->stat.starttime, root->stat.comm,
                                         counters_from(*root)};
    events.push_back(ProcessEvent{EventKind::Started, config_.root_pid, root->stat.ppid,
                                  root->stat.starttime, root->stat.comm,
                                  read_cmdline(config_.root_pid, root->stat.comm), 0});
    discover(list_pids(config_.proc_root), false, events);

    FileRead system = read_file(config_.proc_root + "/stat");
    if (system.status == ReadStatus::Ok) previous_system_ = parse_system_stat(system.content);
    previous_time_ = now;
    return events;
}

void Sampler::discover(const std::vector<int>& pids, bool newborn,
                       std::vector<ProcessEvent>& events) {
    std::unordered_set<int> present(pids.begin(), pids.end());
    for (auto it = outsiders_.begin(); it != outsiders_.end();) {
        it = present.count(*it) ? std::next(it) : outsiders_.erase(it);
    }

    std::vector<PidLink> candidates;
    std::unordered_map<int, StatInfo> candidate_stats;
    for (int pid : pids) {
        if (tracked_.count(pid) || outsiders_.count(pid)) continue;
        auto stat = read_stat(pid);
        if (!stat) continue;
        candidates.push_back(PidLink{pid, stat->ppid});
        candidate_stats.emplace(pid, std::move(*stat));
    }

    std::unordered_set<int> tracked_pids;
    for (const auto& entry : tracked_) tracked_pids.insert(entry.first);
    std::vector<int> adopted = find_new_descendants(tracked_pids, candidates);
    std::unordered_set<int> adopted_set(adopted.begin(), adopted.end());

    for (int pid : adopted) {
        const StatInfo& stat = candidate_stats.at(pid);
        if (is_dead_state(stat.state)) continue;
        auto observation = observe(pid);
        if (!observation) continue;
        ProcessCounters counters = counters_from(*observation);
        if (newborn) counters = zero_counters_like(counters);
        tracked_[pid] = Tracked{stat.ppid, stat.starttime, stat.comm, counters};
        events.push_back(ProcessEvent{EventKind::Started, pid, stat.ppid, stat.starttime,
                                      stat.comm, read_cmdline(pid, stat.comm), 0});
    }
    for (const auto& entry : candidate_stats) {
        if (!adopted_set.count(entry.first)) outsiders_.insert(entry.first);
    }
}

ProcessSample Sampler::measure(int pid, const Observation& observation,
                               const ProcessCounters& previous, double dt) const {
    const double hz = static_cast<double>(config_.clock_ticks_per_second);
    ProcessCounters current = counters_from(observation);

    ProcessSample sample;
    sample.pid = pid;
    sample.ppid = observation.stat.ppid;
    sample.comm = observation.stat.comm;
    sample.state = observation.stat.state;
    sample.threads = observation.stat.num_threads;
    sample.cpu_seconds = static_cast<double>(current.utime + current.stime) / hz;

    if (dt > 0) {
        double user_ticks = static_cast<double>(saturating_delta(current.utime, previous.utime));
        double system_ticks = static_cast<double>(saturating_delta(current.stime, previous.stime));
        sample.user_pct = user_ticks / hz / dt * 100.0;
        sample.system_pct = system_ticks / hz / dt * 100.0;
        sample.cpu_pct = sample.user_pct + sample.system_pct;
        sample.minflt_per_s = static_cast<double>(saturating_delta(current.minflt, previous.minflt)) / dt;
        sample.majflt_per_s = static_cast<double>(saturating_delta(current.majflt, previous.majflt)) / dt;
    }

    if (observation.status && observation.status->vm_rss_kb) {
        sample.rss_kb = *observation.status->vm_rss_kb;
    } else {
        int64_t pages = std::max<int64_t>(observation.stat.rss_pages, 0);
        sample.rss_kb = static_cast<uint64_t>(pages) * static_cast<uint64_t>(config_.page_size) / 1024;
    }

    sample.voluntary_switches_per_s = rate(current.voluntary_switches, previous.voluntary_switches, dt);
    sample.nonvoluntary_switches_per_s =
        rate(current.nonvoluntary_switches, previous.nonvoluntary_switches, dt);
    sample.read_bytes = current.read_bytes;
    sample.write_bytes = current.write_bytes;
    sample.read_bytes_per_s = rate(current.read_bytes, previous.read_bytes, dt);
    sample.write_bytes_per_s = rate(current.write_bytes, previous.write_bytes, dt);
    if (auto blkio = rate(current.blkio_ticks, previous.blkio_ticks, dt)) {
        sample.blkio_wait_pct = *blkio / hz * 100.0;
    }

    if (!observation.status) sample.unavailable.push_back("status");
    if (!observation.io) sample.unavailable.push_back("io");
    if (!config_.delayacct_enabled) sample.unavailable.push_back("blkio_delay");
    return sample;
}

SystemSample Sampler::measure_system(double dt) {
    SystemSample system;
    FileRead stat_read = read_file(config_.proc_root + "/stat");
    std::optional<SystemStat> current;
    if (stat_read.status == ReadStatus::Ok) current = parse_system_stat(stat_read.content);

    if (current) {
        system.cpu_count = current->cpu_count;
        system.procs_running = current->procs_running;
        system.procs_blocked = current->procs_blocked;
        if (previous_system_ && dt > 0) {
            const CpuTimes& now_cpu = current->cpu;
            const CpuTimes& before_cpu = previous_system_->cpu;
            double total = static_cast<double>(saturating_delta(now_cpu.total(), before_cpu.total()));
            double busy = static_cast<double>(saturating_delta(now_cpu.busy(), before_cpu.busy()));
            double iowait = static_cast<double>(saturating_delta(now_cpu.iowait, before_cpu.iowait));
            if (total > 0) {
                system.cpu_busy_pct = busy / total * 100.0;
                system.cpu_iowait_pct = iowait / total * 100.0;
            }
            system.cores_busy = busy / static_cast<double>(config_.clock_ticks_per_second) / dt;
            system.context_switches_per_s = rate(current->ctxt, previous_system_->ctxt, dt);
        }
        previous_system_ = current;
    }

    FileRead meminfo_read = read_file(config_.proc_root + "/meminfo");
    if (meminfo_read.status == ReadStatus::Ok) {
        MemInfo meminfo = parse_meminfo(meminfo_read.content);
        system.mem_total_kb = meminfo.mem_total_kb;
        system.mem_available_kb = meminfo.mem_available_kb;
        if (meminfo.swap_total_kb && meminfo.swap_free_kb) {
            system.swap_used_kb = saturating_delta(*meminfo.swap_total_kb, *meminfo.swap_free_kb);
        }
    }
    return system;
}

Sample Sampler::sample(double now) {
    Sample result;
    result.dt = now - previous_time_;
    previous_time_ = now;
    const double hz = static_cast<double>(config_.clock_ticks_per_second);

    discover(list_pids(config_.proc_root), true, result.events);

    std::vector<int> pids;
    pids.reserve(tracked_.size());
    for (const auto& entry : tracked_) pids.push_back(entry.first);
    std::sort(pids.begin(), pids.end());

    for (int pid : pids) {
        Tracked& tracked = tracked_.at(pid);
        auto observation = observe(pid);
        if (!observation || observation->stat.starttime != tracked.start_ticks) {
            double cpu_seconds =
                static_cast<double>(tracked.counters.utime + tracked.counters.stime) / hz;
            result.events.push_back(ProcessEvent{EventKind::Exited, pid, tracked.ppid,
                                                 tracked.start_ticks, tracked.comm, "", cpu_seconds});
            tracked_.erase(pid);
            continue;
        }

        result.processes.push_back(measure(pid, *observation, tracked.counters, result.dt));
        tracked.counters = counters_from(*observation);
        tracked.ppid = observation->stat.ppid;
        if (observation->stat.comm != tracked.comm) {
            tracked.comm = observation->stat.comm;
            result.events.push_back(ProcessEvent{EventKind::Exec, pid, tracked.ppid, tracked.start_ticks,
                                                 tracked.comm, read_cmdline(pid, tracked.comm), 0});
        }
    }

    result.root_alive = tracked_.count(config_.root_pid) > 0;
    result.system = measure_system(result.dt);
    return result;
}

}
