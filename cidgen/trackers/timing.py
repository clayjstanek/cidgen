"""Section timers for the trackers (`--debug 1`).

    T = Timing(enabled=True)
    with T.cycle("look"):              # one update cycle: per-call times kept for percentiles
        with T("look/gate"):           # a section inside it; "/" nests sections in the report
            ...
    print(T.report(total_s))

Disabled (the default), T(...) and T.cycle(...) return one shared do-nothing context, so the timers
cost about a tenth of a microsecond each and change nothing.

With a log path, every timed block also writes one line to the timing log as it finishes:
    wall_time (ISO date and time, microseconds), sim_time_s, section, dt_ms
and close() writes the summary table next to it (same name, _summary.txt / .csv). The log is flushed to
disk at least every 5 s, so its last line shows how far a long run has got.

To dig into a bottleneck, wrap more of its code in `with self.T("parent/child"):` - no other change.
"""
from __future__ import annotations

import datetime as _dt
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

_NULL = nullcontext()


class _Section:
    __slots__ = ("tm", "name", "t0", "keep")

    def __init__(self, tm, name, keep):
        self.tm, self.name, self.keep = tm, name, keep

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        s = self.tm.stats.get(self.name)
        if s is None:
            s = self.tm.stats[self.name] = [0, 0.0, 0.0]
        s[0] += 1; s[1] += dt
        if dt > s[2]:
            s[2] = dt
        if self.keep:
            self.tm.cycles.setdefault(self.name, []).append(dt)
        f = self.tm.log
        if f is not None:
            now = self.t0 + dt
            wall = self.tm.wall0 + _dt.timedelta(seconds=now - self.tm.perf0)
            f.write(f"{wall.isoformat(timespec='microseconds')},{self.tm.sim_t:.3f},{self.name},{1e3 * dt:.4f}\n")
            if now - self.tm.last_flush > 5.0:              # keep the file current, so progress is visible
                f.flush(); self.tm.last_flush = now
        return False


class Timing:
    def __init__(self, enabled=False, log_path=None):
        self.enabled = bool(enabled)
        self.stats = {}                                    # name -> [calls, total s, max s]
        self.cycles = {}                                   # cycle name -> list of per-call seconds
        self.sim_t = 0.0                                   # simulation time of the current cycle
        self.log = None; self.log_path = None
        self.wall0 = _dt.datetime.now(); self.perf0 = time.perf_counter(); self.last_flush = self.perf0
        if self.enabled and log_path:
            self.log_path = Path(log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = open(self.log_path, "w", buffering=1 << 20)
            self.log.write("wall_time,sim_time_s,section,dt_ms\n")

    @staticmethod
    def stamped_path(folder, stem):
        """<folder>/<stem>_timingLog_YYYYMMDD_HHMMSS.csv"""
        return Path(folder) / f"{stem}_timingLog_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    def close(self, total_s=None, title="tracker timing"):
        """Close the log and write the summary beside it; returns the summary text."""
        text = self.report(total_s, title)
        if self.log is not None:
            self.log.close(); self.log = None
            base = self.log_path.with_suffix("")
            Path(str(base) + "_summary.txt").write_text(text + "\n")
            self.table(total_s).to_csv(str(base) + "_summary.csv", index=False)
        return text

    def __call__(self, name):
        return _Section(self, name, False) if self.enabled else _NULL

    def cycle(self, name):
        return _Section(self, name, True) if self.enabled else _NULL

    def table(self, total_s=None):
        if not self.stats:
            return pd.DataFrame()
        rows = []
        tot = total_s or max(v[1] for v in self.stats.values())
        for name, (n, s, mx) in self.stats.items():
            rows.append(dict(section=name, depth=name.count("/"), calls=n, total_s=s, share=s / tot,
                             mean_ms=1e3 * s / n, max_ms=1e3 * mx))
        df = pd.DataFrame(rows).sort_values("section").reset_index(drop=True)
        for c, v in self.cycles.items():
            a = np.array(v) * 1e3
            k = df.section == c
            df.loc[k, "p50_ms"] = np.percentile(a, 50); df.loc[k, "p95_ms"] = np.percentile(a, 95)
        return df

    def report(self, total_s=None, title="tracker timing"):
        df = self.table(total_s)
        if df.empty:
            return f"{title}: no timing data (run with --debug 1)"
        lines = [f"=== {title}" + (f" (tracker wall time {total_s:.1f} s)" if total_s else "") + " ===",
                 f"{'section':<40} {'calls':>9} {'total s':>9} {'share':>7} {'mean ms':>9} {'p50 ms':>8} "
                 f"{'p95 ms':>8} {'max ms':>9}"]
        for _, r in df.iterrows():
            name = "  " * int(r.depth) + r.section.split("/")[-1]
            p50 = f"{r.p50_ms:8.2f}" if np.isfinite(r.get("p50_ms", np.nan)) else " " * 8
            p95 = f"{r.p95_ms:8.2f}" if np.isfinite(r.get("p95_ms", np.nan)) else " " * 8
            lines.append(f"{name:<40} {int(r.calls):>9,} {r.total_s:>9.2f} {100 * r.share:>6.1f}% "
                         f"{r.mean_ms:>9.3f} {p50} {p95} {r.max_ms:>9.2f}")
        lines.append("share = section total / tracker wall time; p50/p95 per update cycle (look, snapshot)")
        return "\n".join(lines)
