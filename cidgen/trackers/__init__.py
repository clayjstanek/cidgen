"""Multi-target trackers that replace the generator's association-by-label stand-in.

    tracking:
      method: label     # default: v0.6.1 behaviour (reports carry the true object id; one filter per id)
      method: gnn       # global nearest neighbour with track initiation, confirmation and deletion
      method: jpda      # joint probabilistic data association on the same track management
      method: vb        # variational (Turner, Bottone & Avasarala 2014) sliding-window tracker, same management
      method: vbm       # vb, plus merged reports: one report may stand for a set of unresolved tracks
      method: mfa       # Stone Soup multi-frame assignment for confirmed tracks (needs stonesoup + ortools)

Every tracker takes ALL radar reports (clutter included, true ids hidden) and returns
    est      track estimates on the feature grid, with the track number in the 'eid' column
    rep_tid  the track number each radar report updated (-1 = none / never-confirmed track)
    tracks   one row per track (first report, confirmation, last estimate, filter tier)
so features, graph, models and threat prioritisation run unchanged on tracks instead of true objects.
"""
from .gnn import GNNTracker, gnn_settings  # noqa: F401
from .soft import SoftTracker, soft_settings  # noqa: F401


def make_tracker(cfg, method, debug=0, timing_log=None):
    """gnn | jpda | vb | vbm | mfa (Stone Soup).  debug=1 turns on the section timers (timing.py); timing_log is the
    dated per-block log file to write (None: summary only)."""
    if method == "gnn":
        return GNNTracker(cfg, debug, timing_log)
    if method in ("jpda", "vb"):
        return SoftTracker(cfg, method, debug, timing_log)
    if method == "vbm":
        from .merged import MergedVBTracker
        return MergedVBTracker(cfg, debug, timing_log)
    if method == "mfa":
        from .mfa_ss import MFATracker                    # needs stonesoup + ortools
        return MFATracker(cfg, debug, timing_log)
    raise ValueError(f"tracking.method: unknown tracker {method!r}")
