#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

inline std::string fixture(const std::string& name) {
    std::ifstream file(std::string(MLPLAT_FIXTURE_DIR) + "/" + name, std::ios::binary);
    std::ostringstream content;
    content << file.rdbuf();
    return content.str();
}

class TempDir {
public:
    TempDir() {
        static std::atomic<int> counter{0};
        path_ = std::filesystem::temp_directory_path() /
                ("mlplat_test_" + std::to_string(getpid()) + "_" + std::to_string(counter++));
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

struct FakeProcess {
    int pid = 0;
    int ppid = 0;
    std::string comm = "proc";
    char state = 'S';
    uint64_t utime = 0;
    uint64_t stime = 0;
    uint64_t minflt = 0;
    uint64_t majflt = 0;
    uint64_t starttime = 1000;
    int threads = 1;
    int rss_pages = 100;
    uint64_t blkio_ticks = 0;
    uint64_t vm_rss_kb = 400;
    uint64_t voluntary = 0;
    uint64_t nonvoluntary = 0;
    uint64_t read_bytes = 0;
    uint64_t write_bytes = 0;
    std::string cmdline = "proc --flag";
};

class FakeProc {
public:
    FakeProc() { set_system(0, 0, 1000, 0, 100); write_text("meminfo", fixture("meminfo.txt")); }

    std::string root() const { return dir_.path().string(); }

    void set(const FakeProcess& process) {
        std::filesystem::path base = dir_.path() / std::to_string(process.pid);
        std::filesystem::create_directories(base);

        std::vector<std::string> fields(50, "0");
        auto put = [&](int number, const std::string& value) { fields[number - 3] = value; };
        put(3, std::string(1, process.state));
        put(4, std::to_string(process.ppid));
        put(10, std::to_string(process.minflt));
        put(12, std::to_string(process.majflt));
        put(14, std::to_string(process.utime));
        put(15, std::to_string(process.stime));
        put(20, std::to_string(process.threads));
        put(22, std::to_string(process.starttime));
        put(24, std::to_string(process.rss_pages));
        put(42, std::to_string(process.blkio_ticks));
        std::string stat = std::to_string(process.pid) + " (" + process.comm + ")";
        for (const std::string& field : fields) stat += " " + field;
        write_text(std::to_string(process.pid) + "/stat", stat + "\n");

        write_text(std::to_string(process.pid) + "/status",
                   "Name:\t" + process.comm + "\nVmHWM:\t" + std::to_string(process.vm_rss_kb) +
                       " kB\nVmRSS:\t" + std::to_string(process.vm_rss_kb) +
                       " kB\nvoluntary_ctxt_switches:\t" + std::to_string(process.voluntary) +
                       "\nnonvoluntary_ctxt_switches:\t" + std::to_string(process.nonvoluntary) + "\n");
        write_text(std::to_string(process.pid) + "/io",
                   "rchar: 0\nwchar: 0\nread_bytes: " + std::to_string(process.read_bytes) +
                       "\nwrite_bytes: " + std::to_string(process.write_bytes) + "\ncancelled_write_bytes: 0\n");
        std::string cmdline = process.cmdline;
        std::replace(cmdline.begin(), cmdline.end(), ' ', '\0');
        write_text(std::to_string(process.pid) + "/cmdline", cmdline + std::string(1, '\0'));
    }

    void remove(int pid) { std::filesystem::remove_all(dir_.path() / std::to_string(pid)); }

    void remove_file(int pid, const std::string& name) {
        std::filesystem::remove(dir_.path() / std::to_string(pid) / name);
    }

    void deny(int pid, const std::string& name) {
        std::filesystem::permissions(dir_.path() / std::to_string(pid) / name, std::filesystem::perms::none);
    }

    void set_system(uint64_t user, uint64_t system, uint64_t idle, uint64_t iowait, uint64_t ctxt) {
        write_text("stat", "cpu  " + std::to_string(user) + " 0 " + std::to_string(system) + " " +
                               std::to_string(idle) + " " + std::to_string(iowait) +
                               " 0 0 0 0 0\ncpu0 0 0 0 0 0 0 0 0 0 0\ncpu1 0 0 0 0 0 0 0 0 0 0\nctxt " +
                               std::to_string(ctxt) + "\nprocs_running 2\nprocs_blocked 1\n");
    }

private:
    void write_text(const std::string& relative, const std::string& content) {
        std::ofstream(dir_.path() / relative, std::ios::binary | std::ios::trunc) << content;
    }

    TempDir dir_;
};
