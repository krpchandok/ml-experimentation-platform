#include <filesystem>
#include <fstream>
#include <unistd.h>

#include "check.h"
#include "fixtures.h"
#include "procfs.h"

using namespace mlplat;

TEST_CASE("parse_stat reads a real python process") {
    auto stat = parse_stat(fixture("stat_python.txt"));
    REQUIRE(stat.has_value());
    CHECK_EQ(stat->pid, 8702);
    CHECK_EQ(stat->comm, std::string("python3"));
    CHECK_EQ(stat->state, 'S');
    CHECK_EQ(stat->ppid, 8695);
    CHECK_EQ(stat->minflt, 1141u);
    CHECK_EQ(stat->num_threads, 1);
    CHECK_EQ(stat->starttime, 656141u);
    CHECK_EQ(stat->rss_pages, 2492);
    REQUIRE(stat->delayacct_blkio_ticks.has_value());
    CHECK_EQ(*stat->delayacct_blkio_ticks, 0u);
}

TEST_CASE("parse_stat maps every field to the right position") {
    auto stat = parse_stat(fixture("stat_distinct.txt"));
    REQUIRE(stat.has_value());
    CHECK_EQ(stat->pid, 4242);
    CHECK_EQ(stat->comm, std::string("worker"));
    CHECK_EQ(stat->state, 'R');
    CHECK_EQ(stat->ppid, 4000);
    CHECK_EQ(stat->minflt, 10000u);
    CHECK_EQ(stat->majflt, 12000u);
    CHECK_EQ(stat->utime, 14000u);
    CHECK_EQ(stat->stime, 15000u);
    CHECK_EQ(stat->cutime, 16000);
    CHECK_EQ(stat->cstime, 17000);
    CHECK_EQ(stat->num_threads, 20000);
    CHECK_EQ(stat->starttime, 22000u);
    CHECK_EQ(stat->rss_pages, 24000);
    CHECK_EQ(stat->delayacct_blkio_ticks.value_or(0), 42000u);
}

TEST_CASE("parse_stat handles parentheses and spaces in comm") {
    auto stat = parse_stat(fixture("stat_comm_parens.txt"));
    REQUIRE(stat.has_value());
    CHECK_EQ(stat->comm, std::string("my ) (weird proc"));
    CHECK_EQ(stat->state, 'S');
    CHECK_EQ(stat->ppid, 8695);
    CHECK_EQ(stat->starttime, 656141u);
}

TEST_CASE("parse_stat tolerates kernels without delay accounting field") {
    auto stat = parse_stat(fixture("stat_no_delayacct.txt"));
    REQUIRE(stat.has_value());
    CHECK(!stat->delayacct_blkio_ticks.has_value());
    CHECK_EQ(stat->starttime, 656141u);
}

TEST_CASE("parse_stat rejects truncated or malformed content") {
    CHECK(!parse_stat(fixture("stat_truncated.txt")).has_value());
    CHECK(!parse_stat("").has_value());
    CHECK(!parse_stat("not a stat line").has_value());
    CHECK(!parse_stat("12 (x) S abc 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1").has_value());
}

TEST_CASE("parse_status reads memory and context switches") {
    StatusInfo status = parse_status(fixture("status_python.txt"));
    CHECK_EQ(status.vm_rss_kb.value_or(0), 10276u);
    CHECK_EQ(status.vm_hwm_kb.value_or(0), 10276u);
    CHECK_EQ(status.voluntary_ctxt_switches.value_or(99), 2u);
    CHECK_EQ(status.nonvoluntary_ctxt_switches.value_or(99), 0u);
}

TEST_CASE("parse_status reports missing memory fields as absent") {
    StatusInfo status = parse_status(fixture("status_no_memory.txt"));
    CHECK(!status.vm_rss_kb.has_value());
    CHECK(!status.vm_hwm_kb.has_value());
    CHECK(status.voluntary_ctxt_switches.has_value());
}

TEST_CASE("parse_io reads byte counters") {
    IoInfo io = parse_io(fixture("io_python.txt"));
    CHECK_EQ(io.rchar.value_or(0), 319503u);
    CHECK_EQ(io.read_bytes.value_or(0), 135168u);
    CHECK_EQ(io.write_bytes.value_or(99), 0u);
    CHECK_EQ(io.cancelled_write_bytes.value_or(99), 0u);
}

TEST_CASE("parse_io reports missing counters as absent") {
    IoInfo io = parse_io(fixture("io_partial.txt"));
    CHECK(io.rchar.has_value());
    CHECK(!io.read_bytes.has_value());
    CHECK(!io.write_bytes.has_value());
}

TEST_CASE("find_field matches whole keys only") {
    std::string content = "SwapCached:  5 kB\nCached:  7 kB\nVmRSSx: 3\n";
    CHECK_EQ(find_field(content, "Cached").value_or(0), 7u);
    CHECK(!find_field(content, "VmRSS").has_value());
    CHECK(!find_field(content, "Missing").has_value());
}

TEST_CASE("parse_cmdline joins NUL separated arguments") {
    CHECK_EQ(parse_cmdline(fixture("cmdline_python.bin")), std::string("python3 -c import time; time.sleep(30)"));
    CHECK_EQ(parse_cmdline(std::string_view("", 0)), std::string(""));
    CHECK_EQ(parse_cmdline(std::string_view("abcdef\0", 7), 3), std::string("abc"));
}

TEST_CASE("parse_system_stat reads aggregate cpu and counters") {
    auto stat = parse_system_stat(fixture("system_stat.txt"));
    REQUIRE(stat.has_value());
    CHECK_EQ(stat->cpu.user, 4253u);
    CHECK_EQ(stat->cpu.nice, 2176u);
    CHECK_EQ(stat->cpu.system, 4020u);
    CHECK_EQ(stat->cpu.idle, 18357711u);
    CHECK_EQ(stat->cpu.iowait, 634u);
    CHECK_EQ(stat->cpu.softirq, 957u);
    CHECK_EQ(stat->cpu_count, 28);
    CHECK_EQ(stat->ctxt.value_or(0), 3348203u);
    CHECK_EQ(stat->procs_running.value_or(0), 4u);
    CHECK_EQ(stat->procs_blocked.value_or(99), 0u);
    CHECK_EQ(stat->cpu.busy(), 4253u + 2176u + 4020u + 957u);
}

TEST_CASE("parse_system_stat handles old kernels with fewer cpu columns") {
    auto stat = parse_system_stat("cpu  10 0 5 100\ncpu0 10 0 5 100\n");
    REQUIRE(stat.has_value());
    CHECK_EQ(stat->cpu.idle, 100u);
    CHECK_EQ(stat->cpu.iowait, 0u);
    CHECK_EQ(stat->cpu_count, 1);
    CHECK(!stat->ctxt.has_value());
    CHECK(!parse_system_stat("intr 1 2 3\n").has_value());
}

TEST_CASE("parse_meminfo reads memory totals") {
    MemInfo info = parse_meminfo(fixture("meminfo.txt"));
    CHECK_EQ(info.mem_total_kb.value_or(0), 16288444u);
    CHECK_EQ(info.mem_available_kb.value_or(0), 15534620u);
    CHECK_EQ(info.cached_kb.value_or(0), 718304u);
    CHECK_EQ(info.swap_total_kb.value_or(0), 4194304u);
    CHECK_EQ(info.swap_free_kb.value_or(0), 4194304u);
}

TEST_CASE("read_file distinguishes missing, denied and readable files") {
    CHECK(read_file("/proc/self/stat").status == ReadStatus::Ok);
    CHECK(read_file("/proc/self/stat").content.size() > 0);
    CHECK(read_file("/definitely/not/here").status == ReadStatus::NotFound);

    if (geteuid() == 0) return;
    TempDir dir;
    std::filesystem::path secret = dir.path() / "secret";
    std::ofstream(secret) << "hidden";
    std::filesystem::permissions(secret, std::filesystem::perms::none);
    CHECK(read_file(secret.string()).status == ReadStatus::PermissionDenied);
}

TEST_CASE("list_pids returns numeric entries only") {
    TempDir dir;
    for (const char* name : {"1", "42", "self", "sys", "7x"}) {
        std::filesystem::create_directory(dir.path() / name);
    }
    auto pids = list_pids(dir.path().string());
    std::sort(pids.begin(), pids.end());
    REQUIRE(pids.size() == 2);
    CHECK_EQ(pids[0], 1);
    CHECK_EQ(pids[1], 42);
    CHECK(list_pids("/definitely/not/here").empty());
}
