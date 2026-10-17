"""Prepare persistent serial access, then replace this process with the backend."""
from __future__ import annotations

import grp
import os
from pathlib import Path
import pwd
import shlex
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    backend = [sys.executable, str(PROJECT_ROOT / "run.py")]
    if os.getuid() == 0:
        os.execv(backend[0], backend)
        return

    account = pwd.getpwuid(os.getuid()).pw_name
    try:
        dialout = grp.getgrnam("dialout")
    except KeyError:
        raise SystemExit("Linux serial group 'dialout' is missing; install the Ubuntu device packages first.")

    active_groups = {*os.getgroups(), os.getgid()}
    if dialout.gr_gid in active_groups:
        os.execv(backend[0], backend)
        return

    # Account membership persists across reboots; the current desktop session
    # may still have its old group list. sg activates the group for this backend.
    group_launcher = shutil.which("sg")
    if not group_launcher:
        raise SystemExit("Missing sg; install the Ubuntu 'login' package and retry.")
    if account not in dialout.gr_mem:
        sudo = shutil.which("sudo")
        usermod = shutil.which("usermod") or "/usr/sbin/usermod"
        if not sudo or not Path(usermod).is_file():
            raise SystemExit("Serial access setup requires sudo and usermod on Ubuntu.")
        print(
            f"Setting up permanent USB scale access for {account}. "
            "Ubuntu may ask for your password once.",
            flush=True,
        )
        try:
            subprocess.run([sudo, usermod, "-aG", "dialout", account], check=True)
        except subprocess.CalledProcessError:
            raise SystemExit("USB access setup did not complete. Start the backend again to retry.")
        if account not in grp.getgrnam("dialout").gr_mem:
            raise SystemExit("Ubuntu did not save dialout membership; USB access setup is incomplete.")

    # exec preserves the launcher's inherited backend lock. sg changes only this
    # process's group; camera, model loading, and the web app run as the user.
    os.execv(group_launcher, [group_launcher, "dialout", "-c", "exec " + shlex.join(backend)])


if __name__ == "__main__":
    main()
