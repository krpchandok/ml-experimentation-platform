#include "records.h"

#include "json_writer.h"

namespace mlplat {

namespace {

constexpr int kSchemaVersion = 1;
constexpr const char* kAgentVersion = "0.1.0";

const char* event_type(EventKind kind) {
    switch (kind) {
        case EventKind::Started: return "process";
        case EventKind::Exec: return "exec";
        case EventKind::Exited: return "exit";
    }
    return "unknown";
}

std::optional<double> sum_present(const std::vector<ProcessSample>& processes,
                                  std::optional<double> ProcessSample::*member) {
    std::optional<double> total;
    for (const ProcessSample& process : processes) {
        const auto& value = process.*member;
        if (value) total = total.value_or(0) + *value;
    }
    return total;
}

void write_process(JsonWriter& json, const ProcessSample& process) {
    json.begin_object()
        .field("pid", process.pid)
        .field("ppid", process.ppid)
        .field("comm", process.comm)
        .field("state", std::string(1, process.state))
        .field("threads", process.threads)
        .field("cpu_pct", process.cpu_pct)
        .field("user_pct", process.user_pct)
        .field("sys_pct", process.system_pct)
        .field("cpu_s", process.cpu_seconds)
        .field("rss_kb", process.rss_kb)
        .field("minflt_per_s", process.minflt_per_s)
        .field("majflt_per_s", process.majflt_per_s)
        .field("vcsw_per_s", process.voluntary_switches_per_s)
        .field("ivcsw_per_s", process.nonvoluntary_switches_per_s)
        .field("read_bytes", process.read_bytes)
        .field("write_bytes", process.write_bytes)
        .field("read_bytes_per_s", process.read_bytes_per_s)
        .field("write_bytes_per_s", process.write_bytes_per_s)
        .field("blkio_wait_pct", process.blkio_wait_pct);
    if (!process.unavailable.empty()) {
        json.key("unavailable").begin_array();
        for (const std::string& name : process.unavailable) json.value(name);
        json.end_array();
    }
    json.end_object();
}

void write_unavailable(JsonWriter& json, const std::vector<std::string>& names) {
    if (names.empty()) return;
    json.key("unavailable").begin_array();
    for (const std::string& name : names) json.value(name);
    json.end_array();
}

void write_gpu_header(JsonWriter& json, const GpuHeader& gpu) {
    json.key("gpu").begin_object().field("available", gpu.available);
    if (!gpu.available) {
        json.field("reason", gpu.reason).end_object();
        return;
    }
    json.field("driver", gpu.driver_version).field("library", gpu.library).field("init_ms", gpu.init_ms);
    json.key("devices").begin_array();
    for (const GpuDeviceInfo& device : gpu.devices) {
        json.begin_object()
            .field("index", static_cast<uint64_t>(device.index))
            .field("name", device.name)
            .field("uuid", device.uuid)
            .field("mem_total_mb", device.memory_total_mb)
            .end_object();
    }
    json.end_array().end_object();
}

void write_gpu_sample(JsonWriter& json, const std::optional<GpuSample>& gpu) {
    if (!gpu) {
        json.field("gpu", "unavailable");
        return;
    }
    json.key("gpu").begin_object().field("process_info", gpu->process_info);
    json.key("devices").begin_array();
    for (const GpuDeviceSample& device : gpu->devices) {
        json.begin_object()
            .field("index", static_cast<uint64_t>(device.index))
            .field("util_pct", device.util_pct)
            .field("util_source", device.util_source)
            .field("mem_util_pct", device.mem_util_pct)
            .field("mem_used_mb", device.mem_used_mb)
            .field("mem_total_mb", device.mem_total_mb)
            .field("power_w", device.power_w)
            .field("temp_c", device.temp_c);
        write_unavailable(json, device.unavailable);
        json.end_object();
    }
    json.end_array();
    json.key("procs").begin_array();
    for (const GpuProcessSample& process : gpu->processes) {
        json.begin_object()
            .field("pid", process.pid)
            .field("device", static_cast<uint64_t>(process.device))
            .field("mem_used_mb", process.mem_used_mb)
            .end_object();
    }
    json.end_array().end_object();
}

}

std::string format_header(const HeaderInfo& header) {
    JsonWriter json;
    json.begin_object()
        .field("type", "header")
        .field("schema", kSchemaVersion)
        .field("agent_version", kAgentVersion)
        .field("root_pid", header.root_pid)
        .field("interval_ms", header.interval_ms)
        .field("clk_tck", static_cast<int64_t>(header.clock_ticks_per_second))
        .field("page_size", static_cast<int64_t>(header.page_size))
        .field("online_cpus", static_cast<int64_t>(header.online_cpus))
        .field("delayacct", header.delayacct_enabled)
        .field("pidfd", header.pidfd_supported)
        .field("hostname", header.hostname)
        .field("proc_root", header.proc_root)
        .field("t_mono", header.t_mono)
        .field("t_wall", header.t_wall);
    write_gpu_header(json, header.gpu);
    json.end_object();
    return json.str();
}

std::string format_event(const ProcessEvent& event, double t_mono) {
    JsonWriter json;
    json.begin_object()
        .field("type", event_type(event.kind))
        .field("t_mono", t_mono)
        .field("pid", event.pid)
        .field("ppid", event.ppid)
        .field("start_ticks", event.start_ticks)
        .field("comm", event.comm);
    if (event.kind == EventKind::Exited) {
        json.field("cpu_s", event.cpu_seconds);
    } else {
        json.field("cmdline", event.cmdline);
    }
    json.end_object();
    return json.str();
}

std::string format_sample(const Sample& sample, uint64_t seq, double t_mono, double t_wall,
                          const AgentUsage& agent, const std::optional<GpuSample>& gpu) {
    double tree_cpu = 0;
    uint64_t tree_rss = 0;
    for (const ProcessSample& process : sample.processes) {
        tree_cpu += process.cpu_pct;
        tree_rss += process.rss_kb;
    }

    JsonWriter json;
    json.begin_object()
        .field("type", "sample")
        .field("seq", seq)
        .field("t_mono", t_mono)
        .field("t_wall", t_wall)
        .field("dt", sample.dt);

    const SystemSample& system = sample.system;
    json.key("system").begin_object()
        .field("cpu_count", system.cpu_count)
        .field("cpu_busy_pct", system.cpu_busy_pct)
        .field("cpu_iowait_pct", system.cpu_iowait_pct)
        .field("cores_busy", system.cores_busy)
        .field("ctxt_per_s", system.context_switches_per_s)
        .field("procs_running", system.procs_running)
        .field("procs_blocked", system.procs_blocked)
        .field("mem_total_kb", system.mem_total_kb)
        .field("mem_available_kb", system.mem_available_kb)
        .field("swap_used_kb", system.swap_used_kb)
        .end_object();

    json.key("tree").begin_object()
        .field("nproc", static_cast<uint64_t>(sample.processes.size()))
        .field("cpu_pct", tree_cpu)
        .field("rss_kb", tree_rss)
        .field("read_bytes_per_s", sum_present(sample.processes, &ProcessSample::read_bytes_per_s))
        .field("write_bytes_per_s", sum_present(sample.processes, &ProcessSample::write_bytes_per_s))
        .end_object();

    json.key("procs").begin_array();
    for (const ProcessSample& process : sample.processes) write_process(json, process);
    json.end_array();

    write_gpu_sample(json, gpu);

    json.key("agent").begin_object()
        .field("cpu_s", agent.cpu_seconds)
        .field("max_rss_kb", agent.max_rss_kb)
        .field("sample_us", agent.sample_us)
        .end_object();

    json.end_object();
    return json.str();
}

std::string format_end(const EndInfo& end) {
    double cpu_pct = end.agent_wall_seconds > 0 ? end.agent_cpu_seconds / end.agent_wall_seconds * 100.0 : 0;
    JsonWriter json;
    json.begin_object()
        .field("type", "end")
        .field("reason", end.reason)
        .field("samples", end.samples)
        .field("t_mono", end.t_mono)
        .field("t_wall", end.t_wall)
        .field("agent_cpu_s", end.agent_cpu_seconds)
        .field("startup_cpu_s", end.startup_cpu_seconds)
        .field("agent_wall_s", end.agent_wall_seconds)
        .field("agent_cpu_pct", cpu_pct)
        .field("sample_us_mean", end.sample_us_mean)
        .field("sample_us_max", end.sample_us_max)
        .field("agent_max_rss_kb", end.max_rss_kb)
        .field("max_tracked", end.max_tracked)
        .end_object();
    return json.str();
}

}
