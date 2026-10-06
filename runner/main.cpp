#include <cerrno>
#include <cstdio>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: runner <command> [arguments...]\n";
        return 1;
    }

    int pipefd[2];
    if (pipe(pipefd) == -1) {
        perror("pipe");
        return 1;
    }

    pid_t pid = fork();

    if (pid < 0) {
        perror("fork");
        close(pipefd[0]);
        close(pipefd[1]);
        return 1;
    }

    if (pid == 0) {
        // Give this job its own process group.
        if (setpgid(0, 0) == -1) {
            perror("setpgid");
            _exit(126);
        }

        close(pipefd[0]);

        // Send both stdout and stderr through the pipe.
        if (dup2(pipefd[1], STDOUT_FILENO) == -1 ||
            dup2(pipefd[1], STDERR_FILENO) == -1) {
            perror("dup2");
            _exit(126);
        }

        close(pipefd[1]);

        execvp(argv[1], &argv[1]);
        perror("execvp");
        _exit(127);
    }

    // Also set the child's group from the parent.
    if (setpgid(pid, pid) == -1 &&
        errno != EACCES && errno != ESRCH) {
        perror("setpgid parent");
    }

    close(pipefd[1]);

    std::cout << "Started job with PID " << pid << '\n';
    std::cout << "Cancel from another Ubuntu terminal:\n"
              << "kill -TERM -- -" << pid << std::endl;

    char buffer[4096];
    std::string pending;
    bool read_failed = false;

    while (true) {
        ssize_t count = read(pipefd[0], buffer, sizeof(buffer));

        if (count == 0) break;

        if (count < 0) {
            if (errno == EINTR) continue;
            perror("read");
            read_failed = true;
            break;
        }

        pending.append(buffer, static_cast<size_t>(count));

        size_t newline;
        while ((newline = pending.find('\n')) != std::string::npos) {
            std::cout << "[job] " << pending.substr(0, newline)
                      << std::endl;
            pending.erase(0, newline + 1);
        }
    }

    if (!pending.empty()) {
        std::cout << "[job] " << pending << std::endl;
    }

    close(pipefd[0]);

    int status;
    while (waitpid(pid, &status, 0) == -1) {
        if (errno == EINTR) continue;
        perror("waitpid");
        return 1;
    }

    if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        std::cout << "Job exited with code " << code << '\n';
        return read_failed ? 1 : code;
    }

    if (WIFSIGNALED(status)) {
        int signal = WTERMSIG(status);
        std::cout << "Job terminated by signal " << signal << '\n';
        return 128 + signal;
    }

    return 1;
}
