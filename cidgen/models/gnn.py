"""Temporal + heterogeneous graph model for Combat ID (spec Sections 1 and 8).

  per-track GRU over the last `window` feature steps  +  current features
        -> track embedding h
  L x HeteroConv:  GATv2 over (track, near, track) with edge attributes
                   SAGE   over asset <-> track and esm_fix <-> track
                   residual + LayerNorm per node type
  heads: allegiance (3), unresolved-formation flag (2), threat priority (1)

`graph=False` skips the message passing, giving the temporal-only model (option 1 in the plan)
with the same encoder, so the comparison isolates what the graph adds.

`tt_conv="gated"` replaces GATv2 on (track, near, track) with GatedEdgeConv: each neighbour's message
is scaled by its own learned gate in [0, 1]. GATv2's attention weights always sum to one over a track's
neighbours, so a track surrounded only by tracks of another allegiance must still take its messages from
them; gates need not sum to anything, so a track can switch every neighbour off and keep its own evidence.
Requires torch and torch_geometric (core only).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv, MessagePassing, SAGEConv
from torch_geometric.utils import degree

from .gnn_data import EDGE_TYPES, sequence


# ----------------------------------------------------------------------------- dataset
SENSOR_GROUPS = {"iff": ("iff_",), "jem": ("jem_",), "lidar": ("lidar_",), "eoir": ("eoir_",), "esm": ("esm_",),
                 "md": ("rad_md_",)}
NOT_SENSOR = {"iff_in_coverage"}          # geometry, known even when the sensor reports nothing


def sensor_group_columns(cols):
    """Column indices of each sensor family (its values, report age, one-hot codes and has_ flag)."""
    out = {}
    for g, prefixes in SENSOR_GROUPS.items():
        idx = [i for i, c in enumerate(cols) if (c.startswith(prefixes) or c == f"has_{g}") and c not in NOT_SENSOR]
        if idx:
            out[g] = np.array(idx)
    return out


class SnapshotDataset(torch.utils.data.Dataset):
    """One HeteroData per snapshot; features normalised with training-split statistics.

    sensor_dropout > 0 (training only): for each track and each sensor family independently, with that
    probability the family's evidence is removed for the whole history window, exactly as if the sensor
    had never reported. This stops the model relying on any one sensor (spec: modality dropout)."""

    def __init__(self, blobs, norm, window=15, sensor_dropout=0.0, seed=0, drop_features=()):
        self.blobs, self.norm, self.window = blobs, norm, window
        self.index = [(bi, si) for bi, b in enumerate(blobs) for si in range(len(b["snaps"]))]
        self.n_fix = next((len(s["x_fix"][0]) for b in blobs for s in b["snaps"] if len(s["x_fix"])), 20)
        self.p_drop = float(sensor_dropout)
        self.groups = sensor_group_columns(blobs[0]["cols"]) if self.p_drop > 0 else {}
        self.rng = np.random.default_rng(seed)
        cols = blobs[0]["cols"]
        missing = [c for c in drop_features if c not in cols]
        if missing:
            raise ValueError(f"--drop-features: unknown feature(s) {missing}")
        self.drop_idx = np.array([cols.index(c) for c in drop_features], int)   # held at their training mean

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        bi, si = self.index[i]
        b = self.blobs[bi]; s = b["snaps"][si]; nm = self.norm
        raw = b["cube"][s["rows"], s["step"]].astype(np.float32)
        seq = sequence(b, s, self.window)
        if self.p_drop > 0:                      # remove whole sensor families, in raw (pre-normalisation) units
            for g, idx in self.groups.items():
                hit = np.flatnonzero(self.rng.random(len(raw)) < self.p_drop)
                if len(hit):
                    raw[np.ix_(hit, idx)] = 0.0
                    seq[np.ix_(hit, np.arange(seq.shape[1]), idx)] = 0.0
        x = np.clip((raw - nm["x_mean"]) / nm["x_std"], -10, 10)
        seq[..., :-1] = np.clip((seq[..., :-1] - nm["x_mean"]) / nm["x_std"], -10, 10) * seq[..., -1:]  # 0 if not held
        if len(self.drop_idx):                   # removed inputs carry no information in any split
            x[:, self.drop_idx] = 0.0
            seq[:, :, self.drop_idx] = 0.0
        d = HeteroData()
        d["track"].x = torch.from_numpy(np.ascontiguousarray(x, np.float32))
        d["track"].x_seq = torch.from_numpy(np.ascontiguousarray(seq, np.float32))
        d["track"].y = torch.from_numpy(np.asarray(s["y"], np.int64))
        d["track"].y_multi = torch.from_numpy(np.asarray(s["y_multi"], np.int64))
        d["track"].y_threat = torch.from_numpy(np.asarray(s["y_threat"], np.float32))
        d["track"].eid = torch.from_numpy(np.asarray(s["eid"], np.int64))
        d["track"].t = torch.full((len(s["eid"]),), s["t"])
        d["track"].run = torch.full((len(s["eid"]),), bi, dtype=torch.long)
        d["asset"].x = torch.from_numpy(np.ascontiguousarray(s["x_asset"], np.float32))
        fx = np.asarray(s["x_fix"], np.float32).reshape(-1, self.n_fix)
        if "fix" in nm and len(fx):
            fx = (fx - nm["fix"][0]) / nm["fix"][1]
        d["esm_fix"].x = torch.from_numpy(np.ascontiguousarray(fx, np.float32))
        for et in EDGE_TYPES:
            ei, ea = s["edges"][et]
            ea = np.asarray(ea, np.float32)
            if et in nm and len(ea):
                ea = (ea - nm[et][0]) / nm[et][1]
            d[et].edge_index = torch.from_numpy(np.array(ei, np.int64).reshape(2, -1))
            if ea.ndim == 1:
                ea = ea[:, None]
            d[et].edge_attr = torch.from_numpy(np.ascontiguousarray(ea, np.float32))
        return d


# ----------------------------------------------------------------------------- model
class GatedEdgeConv(MessagePassing):
    """Track-track messages scaled by a per-edge gate in [0, 1], averaged over the neighbours.

    message_ij = gate_ij * MLP([h_i, h_j, e_ij]),  gate_ij = sigmoid(MLP([h_i, h_j, e_ij]))
    The gate sees the receiving track's own state, the neighbour's and the edge's relative kinematics
    (and co-movement features when the cache has them), so it can learn to keep formation mates and
    shut out tracks that only pass close by. `last_gate` holds the latest gates for inspection."""

    def __init__(self, hidden, edge_dim, dropout=0.0):
        super().__init__(aggr="add")
        z = 2 * hidden + edge_dim
        self.msg = nn.Sequential(nn.Linear(z, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.gate = nn.Sequential(nn.Linear(z, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.drop = nn.Dropout(dropout)
        self.last_gate = None

    def forward(self, x, edge_index, edge_attr):
        x = x if isinstance(x, tuple) else (x, x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        deg = degree(edge_index[1], num_nodes=x[1].size(0), dtype=out.dtype).clamp(min=1).unsqueeze(-1)
        return out / deg

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], -1)
        g = torch.sigmoid(self.gate(z))
        self.last_gate = g.detach()
        return self.drop(g * self.msg(z))


class CIDGNN(nn.Module):
    def __init__(self, n_feat, n_fix=20, hidden=128, layers=2, heads=4, dropout=0.15, graph=True,
                 tt_edge_dim=10, ta_edge_dim=7, tt_conv="gatv2"):
        super().__init__()
        self.graph = graph
        if tt_conv not in ("gatv2", "gated"):
            raise ValueError(f"tt_conv must be 'gatv2' or 'gated', got {tt_conv!r}")
        self.gru = nn.GRU(n_feat + 1, hidden, batch_first=True)
        self.cur = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.asset_enc = nn.Linear(4, hidden)
        self.fix_enc = nn.Linear(n_fix, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers if graph else 0):
            self.convs.append(HeteroConv({
                ("track", "near", "track"): (GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=tt_edge_dim,
                                                       add_self_loops=True, fill_value="mean", dropout=dropout)
                                             if tt_conv == "gatv2" else GatedEdgeConv(hidden, tt_edge_dim, dropout)),
                ("asset", "threatened_by", "track"): SAGEConv((hidden, hidden), hidden),
                ("esm_fix", "candidate", "track"): SAGEConv((hidden, hidden), hidden),
                ("track", "threatens", "asset"): SAGEConv((hidden, hidden), hidden),
                ("track", "hears", "esm_fix"): SAGEConv((hidden, hidden), hidden),
            }, aggr="sum"))
            self.norms.append(nn.ModuleDict({k: nn.LayerNorm(hidden) for k in ("track", "asset", "esm_fix")}))
        self.drop = nn.Dropout(dropout)
        self.head_id = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.head_multi = nn.Linear(hidden, 2)
        self.head_threat = nn.Linear(hidden, 1)

    def forward(self, d):
        _, hT = self.gru(d["track"].x_seq)
        h = self.fuse(torch.cat([hT[-1], self.cur(d["track"].x)], -1))
        hd = {"track": h, "asset": self.asset_enc(d["asset"].x), "esm_fix": self.fix_enc(d["esm_fix"].x)}
        ea = {("track", "near", "track"): d["track", "near", "track"].edge_attr}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(hd, d.edge_index_dict, edge_attr_dict=ea)
            hd = {k: norm[k](hd[k] + self.drop(F.gelu(out[k]))) if k in out else hd[k] for k in hd}
        h = hd["track"]
        return self.head_id(h), self.head_multi(h), self.head_threat(h).squeeze(-1)


# ----------------------------------------------------------------------------- training
def loss_fn(out, d, w_cls, aux=(0.2, 0.5)):
    logit, lm, lt = out
    y = d["track"].y
    m = y >= 0
    loss = F.cross_entropy(logit[m], y[m], weight=w_cls) if m.any() else logit.sum() * 0
    loss = loss + aux[0] * F.cross_entropy(lm, d["track"].y_multi)
    loss = loss + aux[1] * F.mse_loss(lt, d["track"].y_threat)
    return loss


@torch.no_grad()
def predict(model, loader, device):
    """Collect logits for objects of interest with their (run index, eid, t) keys."""
    model.eval()
    L, Y, R, E, T = [], [], [], [], []
    for d in loader:
        d = d.to(device)
        logit, _, _ = model(d)
        m = d["track"].y >= 0
        L.append(logit[m].float().cpu()); Y.append(d["track"].y[m].cpu())
        R.append(d["track"].run[m].cpu()); E.append(d["track"].eid[m].cpu()); T.append(d["track"].t[m].cpu())
    return torch.cat(L), torch.cat(Y), torch.cat(R), torch.cat(E), torch.cat(T)


def fit_temperature(logits, y, log=None):
    """Temperature scaling on validation logits (Guo et al., 2017).

    LBFGS can diverge to NaN on tiny or degenerate validation sets, and clipping NaN leaves NaN, which
    would make every probability NaN. The first attempt is the original fit (so working fits give the
    same temperature as before); if it is not finite, retry with a strong-Wolfe line search, and if
    that also fails use T = 1 (no recalibration)."""
    def _fit(line_search):
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100, line_search_fn=line_search)

        def closure():
            opt.zero_grad()
            l = F.cross_entropy(logits / logT.exp(), y)
            l.backward()
            return l
        opt.step(closure)
        return float(logT.detach().exp().item())

    T = _fit(None)
    if not np.isfinite(T):
        T = _fit("strong_wolfe")
        if log:
            log(f"temperature fit diverged; strong-Wolfe retry gave {T:.3f}")
    if not np.isfinite(T):
        T = 1.0
        if log:
            log("temperature fit failed twice; using T = 1 (uncalibrated)")
    # keep T in a sane range: tiny or degenerate validation sets can otherwise drive it to extremes
    return float(np.clip(T, 0.25, 10.0))


def train(model, train_loader, val_loader, device, w_cls, epochs=30, lr=1e-3, weight_decay=1e-4, patience=6, log=print):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = torch.tensor(w_cls, dtype=torch.float32, device=device)
    best, best_state, bad = np.inf, None, 0
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0; n = 0
        for d in train_loader:
            d = d.to(device)
            opt.zero_grad()
            loss = loss_fn(model(d), d, w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.detach().item() * d.num_graphs; n += d.num_graphs
        sched.step()
        lv, yv, *_ = predict(model, val_loader, device)
        vl = float(F.cross_entropy(lv, yv))
        acc = float((lv.argmax(1) == yv).float().mean())
        log(f"epoch {ep:3d}  train loss {tot / max(n, 1):.4f}  val log loss {vl:.4f}  val acc {acc:.3f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model
