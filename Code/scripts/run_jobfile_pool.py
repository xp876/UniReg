#!/usr/bin/env python
"""Run a text jobfile (one command per line) with a local worker pool.

This is designed for HPC single-node runs where spawning login shells (bash -lc)
can reset PATH/conda and break imports (e.g., torch). Here we:

* Read commands line-by-line
* Execute each command via subprocess with the current environment
* Run up to --jobs commands concurrently
* Write per-job stdout/stderr logs to --log_dir

The jobfile may include shell syntax (conditionals, pipes, env assignments).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JobResult:
    idx: int
    cmd: str
    returncode: int
    log_path: Path


def _run_one(idx: int, cmd: str, log_dir: Path) -> JobResult:
    log_path = log_dir / f"job_{idx:04d}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[job {idx}] cwd={os.getcwd()}\n")
        f.write(f"[job {idx}] cmd={cmd}\n\n")
        f.flush()
        try:
            # Commands in generated jobfiles may include shell syntax
            # (e.g., if [[ -f ... ]]; then ...; fi). Execute through bash
            # while preserving the current environment (conda/PYTHONPATH).
            p = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
                env=os.environ.copy(),
                shell=True,
                executable="/bin/bash",
            )
            rc = int(p.returncode)
        except FileNotFoundError as e:
            f.write(f"\nERROR: executable not found: {e}\n")
            rc = 127
        except Exception as e:
            f.write(f"\nERROR: unexpected exception: {type(e).__name__}: {e}\n")
            rc = 1
    return JobResult(idx=idx, cmd=cmd, returncode=rc, log_path=log_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobfile", required=True, help="Path to jobfile (one command per line)")
    ap.add_argument("--jobs", type=int, default=4, help="Max concurrent jobs")
    ap.add_argument("--log_dir", required=True, help="Directory to write per-job logs")
    args = ap.parse_args()

    jobfile = Path(args.jobfile)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not jobfile.exists():
        print(f"ERROR: jobfile not found: {jobfile}", file=sys.stderr)
        return 2

    cmds: list[str] = []
    for line in jobfile.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cmds.append(s)

    if not cmds:
        print(f"[run_jobfile_pool] No commands found in {jobfile}", file=sys.stderr)
        return 0

    print(f"[run_jobfile_pool] jobfile={jobfile} jobs={args.jobs} n_cmds={len(cmds)}", file=sys.stderr)
    failed: list[JobResult] = []

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_run_one, i, cmd, log_dir): (i, cmd) for i, cmd in enumerate(cmds, start=1)}
        for fut in as_completed(futs):
            res = fut.result()
            if res.returncode != 0:
                failed.append(res)
                print(
                    f"[run_jobfile_pool] FAIL job={res.idx} rc={res.returncode} log={res.log_path}",
                    file=sys.stderr,
                )
            else:
                print(f"[run_jobfile_pool] OK   job={res.idx} log={res.log_path}", file=sys.stderr)

    if failed:
        print(f"\n[run_jobfile_pool] {len(failed)} job(s) failed. Showing first 5:", file=sys.stderr)
        for r in failed[:5]:
            print(f"  - job={r.idx} rc={r.returncode} log={r.log_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
