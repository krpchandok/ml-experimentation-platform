#include "procfs.h"

#include <cerrno>
#include <charconv>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

namespace mlplat {

namespace {

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

std::string_view trim(std::string_view text) {
    while (!text.empty() && is_space(text.front())) text.remove_prefix(1);
    while (!text.empty() && is_space(text.back())) text.remove_suffix(1);
    return text;
}

template <typename T>
std::optional<T> to_number(std::string_view token) {
    T value{};
    auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (error != std::errc() || end != token.data() + token.size()) return std::nullopt;
    return value;
}

std::vector<std::string_view> split_whitespace(std::string_view text) {
    std::vector<std::string_view> tokens;
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && is_space(text[i])) ++i;
        std::size_t start = i;
        while (i < text.size() && !is_space(text[i])) ++i;
        if (i > start) tokens.push_back(text.substr(start, i - start));
    }
    return tokens;
}

std::string_view first_token(std::string_view text) {
    text = trim(text);
    std::size_t end = 0;
    while (end < text.size() && !is_space(text[end])) ++end;
    return text.substr(0, end);
}

ReadStatus status_from_errno(int error) {
    switch (error) {
        case ENOENT:
        case ESRCH:
            return ReadStatus::NotFound;
        case EACCES:
        case EPERM:
            return ReadStatus::PermissionDenied;
        default:
            return ReadStatus::Error;
    }
}

}

FileRead read_file(const std::string& path) {
    FileRead result;
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        result.status = status_from_errno(errno);
        return result;
    }
    char buffer[4096];
    while (true) {
        ssize_t count = ::read(fd, buffer, sizeof(buffer));
        if (count > 0) {
            result.content.append(buffer, static_cast<std::size_t>(count));
            continue;
        }
        if (count == 0) {
            result.status = ReadStatus::Ok;
            break;
        }
        if (errno == EINTR) continue;
        result.status = status_from_errno(errno);
        result.content.clear();
        break;
    }
    ::close(fd);
    return result;
}

std::optional<StatInfo> parse_stat(std::string_view content) {
    std::size_t open_paren = content.find('(');
    std::size_t close_paren = content.rfind(')');
    if (open_paren == std::string_view::npos || close_paren == std::string_view::npos ||
        close_paren < open_paren) {
        return std::nullopt;
    }

    StatInfo info;
    auto pid = to_number<int>(trim(content.substr(0, open_paren)));
    if (!pid) return std::nullopt;
    info.pid = *pid;
    info.comm = std::string(content.substr(open_paren + 1, close_paren - open_paren - 1));

    auto fields = split_whitespace(content.substr(close_paren + 1));
    constexpr std::size_t first_field_number = 3;
    auto field = [&](std::size_t number) -> std::optional<std::string_view> {
        std::size_t index = number - first_field_number;
        if (index >= fields.size()) return std::nullopt;
        return fields[index];
    };
    auto unsigned_field = [&](std::size_t number) -> std::optional<uint64_t> {
        auto token = field(number);
        return token ? to_number<uint64_t>(*token) : std::nullopt;
    };
    auto signed_field = [&](std::size_t number) -> std::optional<int64_t> {
        auto token = field(number);
        return token ? to_number<int64_t>(*token) : std::nullopt;
    };

    auto state = field(3);
    auto ppid = signed_field(4);
    auto minflt = unsigned_field(10);
    auto majflt = unsigned_field(12);
    auto utime = unsigned_field(14);
    auto stime = unsigned_field(15);
    auto cutime = signed_field(16);
    auto cstime = signed_field(17);
    auto num_threads = signed_field(20);
    auto starttime = unsigned_field(22);
    auto rss = signed_field(24);
    if (!state || state->empty() || !ppid || !minflt || !majflt || !utime || !stime || !cutime ||
        !cstime || !num_threads || !starttime || !rss) {
        return std::nullopt;
    }

    info.state = state->front();
    info.ppid = static_cast<int>(*ppid);
    info.minflt = *minflt;
    info.majflt = *majflt;
    info.utime = *utime;
    info.stime = *stime;
    info.cutime = *cutime;
    info.cstime = *cstime;
    info.num_threads = *num_threads;
    info.starttime = *starttime;
    info.rss_pages = *rss;
    info.delayacct_blkio_ticks = unsigned_field(42);
    return info;
}

std::optional<uint64_t> find_field(std::string_view content, std::string_view key) {
    std::size_t position = 0;
    while (position < content.size()) {
        std::size_t line_end = content.find('\n', position);
        if (line_end == std::string_view::npos) line_end = content.size();
        std::string_view line = content.substr(position, line_end - position);
        if (line.size() > key.size() && line.compare(0, key.size(), key) == 0 &&
            line[key.size()] == ':') {
            return to_number<uint64_t>(first_token(line.substr(key.size() + 1)));
        }
        position = line_end + 1;
    }
    return std::nullopt;
}

StatusInfo parse_status(std::string_view content) {
    StatusInfo info;
    info.vm_rss_kb = find_field(content, "VmRSS");
    info.vm_hwm_kb = find_field(content, "VmHWM");
    info.voluntary_ctxt_switches = find_field(content, "voluntary_ctxt_switches");
    info.nonvoluntary_ctxt_switches = find_field(content, "nonvoluntary_ctxt_switches");
    return info;
}

IoInfo parse_io(std::string_view content) {
    IoInfo info;
    info.rchar = find_field(content, "rchar");
    info.wchar = find_field(content, "wchar");
    info.read_bytes = find_field(content, "read_bytes");
    info.write_bytes = find_field(content, "write_bytes");
    info.cancelled_write_bytes = find_field(content, "cancelled_write_bytes");
    return info;
}

std::string parse_cmdline(std::string_view content, std::size_t max_length) {
    while (!content.empty() && content.back() == '\0') content.remove_suffix(1);
    std::string result(content.substr(0, max_length));
    for (char& c : result) {
        if (c == '\0') c = ' ';
    }
    return result;
}

std::optional<SystemStat> parse_system_stat(std::string_view content) {
    SystemStat stat;
    bool found_cpu = false;
    std::size_t position = 0;
    while (position < content.size()) {
        std::size_t line_end = content.find('\n', position);
        if (line_end == std::string_view::npos) line_end = content.size();
        std::string_view line = content.substr(position, line_end - position);
        position = line_end + 1;

        auto tokens = split_whitespace(line);
        if (tokens.empty()) continue;
        std::string_view name = tokens[0];

        if (name == "cpu") {
            uint64_t* targets[] = {&stat.cpu.user,   &stat.cpu.nice,   &stat.cpu.system,
                                   &stat.cpu.idle,   &stat.cpu.iowait, &stat.cpu.irq,
                                   &stat.cpu.softirq, &stat.cpu.steal};
            std::size_t available = tokens.size() - 1;
            if (available < 4) return std::nullopt;
            for (std::size_t i = 0; i < std::size(targets) && i < available; ++i) {
                auto value = to_number<uint64_t>(tokens[i + 1]);
                if (!value) return std::nullopt;
                *targets[i] = *value;
            }
            found_cpu = true;
        } else if (name.size() > 3 && name.compare(0, 3, "cpu") == 0) {
            ++stat.cpu_count;
        } else if (tokens.size() >= 2 && name == "ctxt") {
            stat.ctxt = to_number<uint64_t>(tokens[1]);
        } else if (tokens.size() >= 2 && name == "procs_running") {
            stat.procs_running = to_number<uint64_t>(tokens[1]);
        } else if (tokens.size() >= 2 && name == "procs_blocked") {
            stat.procs_blocked = to_number<uint64_t>(tokens[1]);
        }
    }
    if (!found_cpu) return std::nullopt;
    return stat;
}

MemInfo parse_meminfo(std::string_view content) {
    MemInfo info;
    info.mem_total_kb = find_field(content, "MemTotal");
    info.mem_free_kb = find_field(content, "MemFree");
    info.mem_available_kb = find_field(content, "MemAvailable");
    info.cached_kb = find_field(content, "Cached");
    info.swap_total_kb = find_field(content, "SwapTotal");
    info.swap_free_kb = find_field(content, "SwapFree");
    return info;
}

std::vector<int> list_pids(const std::string& proc_root) {
    std::vector<int> pids;
    DIR* directory = ::opendir(proc_root.c_str());
    if (directory == nullptr) return pids;
    while (dirent* entry = ::readdir(directory)) {
        std::string_view name(entry->d_name);
        if (name.empty() || name[0] < '0' || name[0] > '9') continue;
        if (auto pid = to_number<int>(name)) pids.push_back(*pid);
    }
    ::closedir(directory);
    return pids;
}

}
