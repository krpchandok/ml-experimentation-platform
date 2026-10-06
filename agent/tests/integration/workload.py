import argparse
import json
import os
import subprocess
import sys
import time

CHUNK = b"x" * (1024 * 1024)


def burn_cpu(seconds):
    deadline = time.monotonic() + seconds
    value = 0
    while time.monotonic() < deadline:
        for i in range(10000):
            value += i * i
    return value


def heavy_io(seconds, directory):
    path = os.path.join(directory, f"io_{os.getpid()}.bin")
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            with open(path, "wb") as handle:
                for _ in range(16):
                    handle.write(CHUNK)
                handle.flush()
                os.fsync(handle.fileno())
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                while os.read(fd, len(CHUNK)):
                    pass
            finally:
                os.close(fd)
    finally:
        if os.path.exists(path):
            os.remove(path)


def spawn(role, seconds, directory):
    command = [sys.executable, __file__, "--role", role, "--seconds", str(seconds), "--dir", directory]
    return subprocess.Popen(command)


def run_root(seconds, directory, pid_file, extra_idle):
    children = {
        "cpu": spawn("cpu", seconds, directory),
        "io": spawn("io", seconds, directory),
        "spawner": spawn("spawner", seconds, directory),
    }
    idle = [spawn("idle", seconds, directory) for _ in range(extra_idle)]
    roles = {"root": os.getpid(), **{name: child.pid for name, child in children.items()}}
    roles["idle"] = [child.pid for child in idle]
    with open(pid_file, "w") as handle:
        json.dump(roles, handle)
    for child in [*children.values(), *idle]:
        child.wait()


def run_spawner(seconds, directory):
    grandchild = spawn("cpu", seconds, directory)
    with open(os.path.join(directory, "grandchild.pid"), "w") as handle:
        handle.write(str(grandchild.pid))
    grandchild.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="root")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--pid-file")
    parser.add_argument("--extra-idle", type=int, default=0)
    args = parser.parse_args()

    if args.role == "root":
        run_root(args.seconds, args.dir, args.pid_file, args.extra_idle)
    elif args.role == "cpu":
        burn_cpu(args.seconds)
    elif args.role == "io":
        heavy_io(args.seconds, args.dir)
    elif args.role == "spawner":
        run_spawner(args.seconds, args.dir)
    elif args.role == "idle":
        time.sleep(args.seconds)


if __name__ == "__main__":
    main()
