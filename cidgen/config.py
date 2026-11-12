"""Scenario configuration: defaults.yaml + a scenario overlay, deep-merged."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "configs"


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def find_scenario(name_or_path: str) -> Path:
    """Resolve a scenario name. configs/scenarios is searched first (so "S0" is always the baseline),
    then subfolders such as configs/scenarios/transfer, where only an exact file name matches."""
    p = Path(name_or_path)
    if p.exists():
        return p
    d = CONFIG_DIR / "scenarios"
    # exact name, then "<ID>_..." (so S1 never matches S11), then any prefix
    for pattern in (f"{name_or_path}.yaml", f"{name_or_path}_*.yaml", f"{name_or_path}*.yaml"):
        hits = sorted(d.glob(pattern))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(f"'{name_or_path}' is ambiguous: {[h.stem for h in hits]}")
    hits = sorted(d.glob(f"*/{name_or_path}.yaml"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(f"'{name_or_path}' is ambiguous: {[str(h.relative_to(d)) for h in hits]}")
    raise FileNotFoundError(f"No scenario matching '{name_or_path}' in {d}")


def _overlay(scenario: str, depth=0) -> tuple[dict, dict]:
    """A scenario's overrides, with any `base:` scenario's overrides applied first."""
    if depth > 5:
        raise ValueError("scenario base chain too deep")
    path = find_scenario(scenario)
    with open(path) as f:
        sc = yaml.safe_load(f) or {}
    over = {}
    if sc.get("base"):
        base_over, _ = _overlay(sc["base"], depth + 1)
        over = _deep_merge(over, base_over)
    over = _deep_merge(over, sc.get("overrides", {}))
    return over, dict(sc, _path=str(path))


def load_config(scenario: str, extra_overrides: dict | None = None) -> dict:
    """Load defaults, apply the scenario overlay (and its base, if any), then any extra overrides."""
    with open(CONFIG_DIR / "defaults.yaml") as f:
        cfg = yaml.safe_load(f)
    over, sc = _overlay(scenario)
    cfg = _deep_merge(cfg, over)
    if extra_overrides:
        cfg = _deep_merge(cfg, extra_overrides)
    cfg["scenario_name"] = sc.get("name", Path(sc["_path"]).stem)
    if sc.get("randomize"):
        cfg["randomize"] = sc["randomize"]
    return cfg


def apply_randomize(cfg: dict, seed: int) -> dict:
    """Per-run draws for a `randomize:` block, e.g. {"sensors.radar_noise_scale": [0.8, 1.8]} (uniform),
    {"composition.n_hostile_groups": {"int": [1, 2]}} or {"x.y": {"choice": [a, b]}}. Uses its own
    random stream, so the rest of the scenario's randomness is unchanged. Draws are recorded in
    cfg["randomized_values"] and therefore in config_resolved.yaml and scenario_meta.json."""
    spec = cfg.get("randomize")
    if not spec:
        return cfg
    import numpy as np
    rng = np.random.default_rng([int(seed), 104729])
    cfg = copy.deepcopy(cfg)
    drawn = {}
    for key in sorted(spec):
        rule = spec[key]
        if isinstance(rule, dict) and "int" in rule:
            v = int(rng.integers(rule["int"][0], rule["int"][1] + 1))
        elif isinstance(rule, dict) and "choice" in rule:
            v = rule["choice"][int(rng.integers(len(rule["choice"])))]
        else:
            v = float(rng.uniform(rule[0], rule[1]))
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"randomize: '{key}' is not a parameter in the configuration")
        node[parts[-1]] = v
        drawn[key] = v
    cfg["randomized_values"] = drawn
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
