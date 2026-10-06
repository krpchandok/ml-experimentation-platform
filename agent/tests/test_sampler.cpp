#include <algorithm>
#include <unistd.h>

#include "check.h"
#include "fixtures.h"
#include "sampler.h"

using namespace mlplat;

namespace {

SamplerConfig config_for(const FakeProc& proc, int root_pid, bool delayacct = false) {
    SamplerConfig config;
    config.proc_root = proc.root();
    config.root_pid = root_pid;
    config.clock_ticks_per_second = 100;
    config.page_size = 4096;
    config.delayacct_enabled = delayacct;
    return config;
}

std::vector<int> event_pids(const std::vector<ProcessEvent>& events, EventKind kind) {
    std::vector<int> pids;
    for (const ProcessEvent& event : events) {
        if (event.kind == kind) pids.push_back(event.pid);
    }
    std::sort(pids.begin(), pids.end());
    return pids;
}

const ProcessSample* find_process(const Sample& sample, int pid) {
    for (const ProcessSample& process : sample.processes) {
        if (process.pid == pid) return &process;
    }
    return nullptr;
}

void build_tree(FakeProc& proc) {
    proc.set({.pid = 100, .ppid = 1, .comm = "python"});
    proc.set({.pid = 101, .ppid = 100, .comm = "worker0"});
    proc.set({.pid = 102, .ppid = 100, .comm = "worker1"});
    proc.set({.pid = 103, .ppid = 101, .comm = "helper"});
    proc.set({.pid = 200, .ppid = 1, .comm = "unrelated"});
    proc.set({.pid = 201, .ppid = 200, .comm = "unrelated_child"});
}

}

TEST_CASE("prime discovers the full tree and ignores unrelated processes") {
    FakeProc proc;
    build_tree(proc);
    Sampler sampler(config_for(proc, 100));
    auto events = sampler.prime(0.0);
    REQUIRE(events.has_value());
    auto started = event_pids(*events, EventKind::Started);
    REQUIRE(started.size() == 4);
    CHECK_EQ(started[0], 100);
    CHECK_EQ(started[3], 103);
    CHECK_EQ(sampler.tracked_count(), 4u);
    for (const ProcessEvent& event : *events) {
        if (event.pid == 101) CHECK_EQ(event.cmdline, std::string("proc --flag"));
    }
}

TEST_CASE("prime fails when the root process does not exist") {
    FakeProc proc;
    Sampler sampler(config_for(proc, 999));
    CHECK(!sampler.prime(0.0).has_value());
}

TEST_CASE("sample computes per-interval rates from counter deltas") {
    FakeProc proc;
    proc.set({.pid = 100, .ppid = 1, .utime = 100, .stime = 20, .minflt = 10, .majflt = 1, .voluntary = 5,
              .nonvoluntary = 1, .read_bytes = 4096, .write_bytes = 0});
    Sampler sampler(config_for(proc, 100));
    REQUIRE(sampler.prime(10.0).has_value());

    proc.set({.pid = 100, .ppid = 1, .utime = 250, .stime = 70, .minflt = 410, .majflt = 5, .voluntary = 105,
              .nonvoluntary = 41, .read_bytes = 4096 + 4000000, .write_bytes = 2000000});
    Sample sample = sampler.sample(12.0);
    CHECK_NEAR(sample.dt, 2.0, 1e-9);
    const ProcessSample* root = find_process(sample, 100);
    REQUIRE(root != nullptr);
    CHECK_NEAR(root->user_pct, 75.0, 1e-6);
    CHECK_NEAR(root->system_pct, 25.0, 1e-6);
    CHECK_NEAR(root->cpu_pct, 100.0, 1e-6);
    CHECK_NEAR(root->cpu_seconds, 3.2, 1e-9);
    CHECK_NEAR(root->minflt_per_s.value_or(-1), 200.0, 1e-6);
    CHECK_NEAR(root->majflt_per_s.value_or(-1), 2.0, 1e-6);
    CHECK_NEAR(root->voluntary_switches_per_s.value_or(-1), 50.0, 1e-6);
    CHECK_NEAR(root->nonvoluntary_switches_per_s.value_or(-1), 20.0, 1e-6);
    CHECK_NEAR(root->read_bytes_per_s.value_or(-1), 2000000.0, 1e-6);
    CHECK_NEAR(root->write_bytes_per_s.value_or(-1), 1000000.0, 1e-6);
    CHECK_EQ(root->rss_kb, 400u);
    CHECK(!root->blkio_wait_pct.has_value());
    CHECK(sample.root_alive);
}

TEST_CASE("blkio wait is reported only when delay accounting is enabled") {
    FakeProc proc;
    proc.set({.pid = 100, .ppid = 1, .blkio_ticks = 10});
    Sampler sampler(config_for(proc, 100, true));
    REQUIRE(sampler.prime(0.0).has_value());
    proc.set({.pid = 100, .ppid = 1, .blkio_ticks = 60});
    Sample sample = sampler.sample(1.0);
    const ProcessSample* root = find_process(sample, 100);
    REQUIRE(root != nullptr);
    CHECK_NEAR(root->blkio_wait_pct.value_or(-1), 50.0, 1e-6);
}

TEST_CASE("processes forked after priming are adopted with a zero baseline") {
    FakeProc proc;
    build_tree(proc);
    Sampler sampler(config_for(proc, 100));
    REQUIRE(sampler.prime(0.0).has_value());

    proc.set({.pid = 104, .ppid = 102, .comm = "late_worker", .utime = 30, .read_bytes = 1000});
    proc.set({.pid = 105, .ppid = 104, .comm = "late_grandchild", .utime = 10});
    Sample sample = sampler.sample(1.0);
    auto started = event_pids(sample.events, EventKind::Started);
    REQUIRE(started.size() == 2);
    CHECK_EQ(started[0], 104);
    CHECK_EQ(started[1], 105);
    const ProcessSample* late = find_process(sample, 104);
    REQUIRE(late != nullptr);
    CHECK_NEAR(late->cpu_pct, 30.0, 1e-6);
    CHECK_NEAR(late->read_bytes_per_s.value_or(-1), 1000.0, 1e-6);
    CHECK(find_process(sample, 201) == nullptr);
}

TEST_CASE("exited and zombie processes produce exit events without crashing") {
    FakeProc proc;
    build_tree(proc);
    Sampler sampler(config_for(proc, 100));
    REQUIRE(sampler.prime(0.0).has_value());

    proc.remove(102);
    proc.set({.pid = 103, .ppid = 101, .comm = "helper", .state = 'Z'});
    Sample sample = sampler.sample(1.0);
    auto exited = event_pids(sample.events, EventKind::Exited);
    REQUIRE(exited.size() == 2);
    CHECK_EQ(exited[0], 102);
    CHECK_EQ(exited[1], 103);
    CHECK(find_process(sample, 102) == nullptr);
    CHECK(find_process(sample, 101) != nullptr);
    CHECK(sample.root_alive);
    CHECK_EQ(sampler.tracked_count(), 2u);
}

TEST_CASE("pid reuse is detected through start time") {
    FakeProc proc;
    build_tree(proc);
    Sampler sampler(config_for(proc, 100));
    REQUIRE(sampler.prime(0.0).has_value());
    proc.set({.pid = 101, .ppid = 100, .comm = "worker0", .starttime = 5000});
    Sample sample = sampler.sample(1.0);
    auto exited = event_pids(sample.events, EventKind::Exited);
    REQUIRE(exited.size() == 1);
    CHECK_EQ(exited[0], 101);
}

TEST_CASE("orphans reparented away from the tree stay tracked") {
    FakeProc proc;
    build_tree(proc);
    Sampler sampler(config_for(proc, 100));
    REQUIRE(sampler.prime(0.0).has_value());
    proc.remove(101);
    proc.set({.pid = 103, .ppid = 1, .comm = "helper", .utime = 50});
    Sample sample = sampler.sample(1.0);
    const ProcessSample* orphan = find_process(sample, 103);
    REQUIRE(orphan != nullptr);
    CHECK_EQ(orphan->ppid, 1);
    CHECK_NEAR(orphan->cpu_pct, 50.0, 1e-6);
}

TEST_CASE("root exit and root zombie end the run") {
    FakeProc proc;
    build_tree(proc);
    Sampler sampler(config_for(proc, 100));
    REQUIRE(sampler.prime(0.0).has_value());
    proc.set({.pid = 100, .ppid = 1, .comm = "python", .state = 'Z'});
    CHECK(!sampler.sample(1.0).root_alive);

    FakeProc gone;
    gone.set({.pid = 100, .ppid = 1});
    Sampler second(config_for(gone, 100));
    REQUIRE(second.prime(0.0).has_value());
    gone.remove(100);
    CHECK(!second.sample(1.0).root_alive);
}

TEST_CASE("unreadable io and missing status are reported as unavailable") {
    if (geteuid() == 0) return;
    FakeProc proc;
    proc.set({.pid = 100, .ppid = 1, .rss_pages = 10, .vm_rss_kb = 999});
    Sampler sampler(config_for(proc, 100));
    REQUIRE(sampler.prime(0.0).has_value());
    proc.deny(100, "io");
    proc.remove_file(100, "status");
    Sample sample = sampler.sample(1.0);
    const ProcessSample* root = find_process(sample, 100);
    REQUIRE(root != nullptr);
    CHECK(!root->read_bytes_per_s.has_value());
    CHECK(!root->voluntary_switches_per_s.has_value());
    CHECK_EQ(root->rss_kb, 40u);
    bool io_flagged = std::count(root->unavailable.begin(), root->unavailable.end(), "io") == 1;
    bool status_flagged = std::count(root->unavailable.begin(), root->unavailable.end(), "status") == 1;
    CHECK(io_flagged);
    CHECK(status_flagged);
}

TEST_CASE("comm change after exec emits an exec event") {
    FakeProc proc;
    proc.set({.pid = 100, .ppid = 1, .comm = "bash"});
    Sampler sampler(config_for(proc, 100));
    REQUIRE(sampler.prime(0.0).has_value());
    proc.set({.pid = 100, .ppid = 1, .comm = "python", .cmdline = "python train.py"});
    Sample sample = sampler.sample(1.0);
    auto execs = event_pids(sample.events, EventKind::Exec);
    REQUIRE(execs.size() == 1);
    for (const ProcessEvent& event : sample.events) {
        if (event.kind == EventKind::Exec) CHECK_EQ(event.cmdline, std::string("python train.py"));
    }
}

TEST_CASE("system sample derives busy and iowait percentages") {
    FakeProc proc;
    proc.set({.pid = 100, .ppid = 1});
    Sampler sampler(config_for(proc, 100));
    REQUIRE(sampler.prime(0.0).has_value());
    proc.set_system(150, 50, 1100, 100, 1100);
    Sample sample = sampler.sample(2.0);
    CHECK_EQ(sample.system.cpu_count, 2);
    CHECK_NEAR(sample.system.cpu_busy_pct.value_or(-1), 50.0, 1e-6);
    CHECK_NEAR(sample.system.cpu_iowait_pct.value_or(-1), 25.0, 1e-6);
    CHECK_NEAR(sample.system.cores_busy.value_or(-1), 1.0, 1e-6);
    CHECK_NEAR(sample.system.context_switches_per_s.value_or(-1), 500.0, 1e-6);
    CHECK_EQ(sample.system.mem_total_kb.value_or(0), 16288444u);
    CHECK_EQ(sample.system.swap_used_kb.value_or(99), 0u);
}
