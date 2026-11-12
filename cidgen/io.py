"""Dataset writer/reader: Parquet when pyarrow (or fastparquet) is installed, CSV otherwise."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def parquet_available() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except ImportError:
            pass
    return False


def write_tables(out_dir: Path, tables: dict, meta: dict, fmt: str = "auto"):
    out_dir.mkdir(parents=True, exist_ok=True)
    use_pq = (fmt == "parquet") or (fmt == "auto" and parquet_available())
    ext = "parquet" if use_pq else "csv"
    for name, df in tables.items():
        path = out_dir / f"{name}.{ext}"
        if use_pq:
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
    meta = dict(meta, format=ext, tables={k: len(v) for k, v in tables.items()})
    (out_dir / "scenario_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return ext


def read_table(run_dir, name: str) -> pd.DataFrame:
    run_dir = Path(run_dir)
    for ext, reader in (("parquet", pd.read_parquet), ("csv", lambda p: pd.read_csv(p, low_memory=False))):
        p = run_dir / f"{name}.{ext}"
        if p.exists():
            return reader(p)
    raise FileNotFoundError(f"{name} not found in {run_dir}")


def read_meta(run_dir) -> dict:
    return json.loads((Path(run_dir) / "scenario_meta.json").read_text())
