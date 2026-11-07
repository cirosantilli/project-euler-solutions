#!/usr/bin/env python3
"""
Run one Lean check at a time, with heap limits and a resident-memory watchdog.

Use inside lake env, passing the Lean executable and its arguments after --.
The inherited lock also protects against a launcher dying before its child.
"""

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

LOCK = Path("/tmp/project-euler-lean.lock")
HEAP_MIB = 2560
RESIDENT_MIB = 3072


def run_lean(command, *, env=None, **kwargs):
    command = [command[0], *(
        argument for argument in command[1:]
        if not argument.startswith(("-M", "-j", "-DElab.async="))
    )]
    command[1:1] = [f"-M{HEAP_MIB}", "-j1", "-DElab.async=false"]
    environment = dict(os.environ if env is None else env, LEAN_NUM_THREADS="1")

    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Lean check holds /tmp/project-euler-lean.lock") from None
        check = kwargs.pop("check", False)
        timeout = kwargs.pop("timeout", 300)
        if kwargs.pop("capture_output", False):
            kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        started = time.monotonic()
        # Lean reserves a large virtual address range. Limit resident pages,
        # rather than rejecting those reservations before elaboration starts.
        with subprocess.Popen(command, env=environment,
                              pass_fds=(lock.fileno(),), **kwargs) as process:
            try:
                while True:
                    try:
                        status = Path(f"/proc/{process.pid}/status").read_text()
                    except FileNotFoundError:
                        status = ""
                    resident = next((int(line.split()[1]) for line in status.splitlines()
                                     if line.startswith("VmRSS:")), 0)
                    if resident > RESIDENT_MIB * 1024:
                        raise MemoryError(f"Lean exceeded {RESIDENT_MIB} MiB resident memory")
                    if timeout is not None and time.monotonic() - started > timeout:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        stdout, stderr = process.communicate(timeout=0.05)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException:
                process.kill()
                process.communicate()
                raise
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            if check:
                result.check_returncode()
            return result


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[:1] == ["--"]:
        arguments.pop(0)
    if not arguments:
        raise SystemExit("usage: lean_checked.py -- lean [arguments]")
    raise SystemExit(run_lean(arguments).returncode)
