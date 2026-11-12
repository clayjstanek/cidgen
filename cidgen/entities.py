"""Entity population and behaviors (spec Section 3): allegiance, class/variant, role, group,
transponder, emitter timeline, deception flags, and truth trajectories."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import FRIEND, HOSTILE, NEUTRAL, NONE
from .geometry import bearing_vec
from .kinematics import PHASES, Waypoint, fly, follow
from .library import EMITTERS, ENGINE_CODES, PLATFORMS
from .world import World


@dataclass
class Entity:
    eid: int
    cls: str
    variant: str
    allegiance: int
    role: str
    group_id: int
    tier: int
    pos: np.ndarray
    vel: np.ndarray
    alive: np.ndarray
    phase: np.ndarray
    is_ooi: bool = True                     # object of interest (birds are not)
    transponder: str = "none"               # none | mode3c | mode5_l1 | mode5_l2 | spoof_mode3c | spoof_replay
    emitter_obs: str | None = None          # observed emitter code (after variant overlap)
    engine_obs: str | None = None           # observed engine / rotor code
    dims: tuple = (0.0, 0.0, 0.0)           # true LIDAR dims (length, span, tail) after variant scale
    emitting: np.ndarray | None = None      # (T,) bool
    target_asset: int = -1
    flags: dict = field(default_factory=dict)
    iff_up: np.ndarray | None = None        # (T,) transponder working; None = always
    rcs_mult: np.ndarray | None = None      # hypersonic (Tier 3): (T,) RCS multiplier; None = 1


class Population:
    """Builds every entity for one scenario run."""

    def __init__(self, cfg: dict, world: World, rng: np.random.Generator):
        self.cfg, self.world, self.rng = cfg, world, rng
        self.T = int(cfg["run"]["horizon_s"] / cfg["run"]["dt_truth_s"]) + 1
        self.entities: list[Entity] = []
        self._gid = 0
        self._ids = list(rng.permutation(np.arange(1000, 1000 + 5000)))   # shuffled, no ordering leak
        self.rel = cfg["relational"]
        self.tiers = set(cfg["run"]["tiers_enabled"])

    # ------------------------------------------------------------------ helpers
    def _new_gid(self):
        self._gid += 1
        return self._gid

    def _threat_point(self, r_lo, r_hi, about=None):
        w = self.cfg["world"]
        hw = w["threat_sector_halfwidth_deg"]
        if self.rng.random() < self.cfg["deception"].get("p_offaxis_ingress", 0.0):
            brg = self.world.threat_axis_deg + self.rng.choice([-1, 1]) * self.rng.uniform(hw + 10, 170)
        else:
            brg = self.world.threat_axis_deg + self.rng.uniform(-hw, hw)
        base = np.zeros(2) if about is None else about
        return base + bearing_vec(brg) * self.rng.uniform(r_lo, r_hi) * 1000.0

    def _signature(self, cls, variant):
        """Observed emitter/engine codes with variant overlap rho, and LIDAR dims."""
        p = PLATFORMS[cls]
        rho = self.cfg["ambiguity"]["rho_variant"]
        obs_var = variant
        if variant in ("blue", "red") and self.rng.random() < rho:
            obs_var = self.rng.choice(["blue", "red"])
        em = p.emitters.get(obs_var, p.emitters.get(variant))
        en = p.engines.get(obs_var, p.engines.get(variant))
        sc = p.dim_scale.get(variant, 1.0) * self.rng.normal(1.0, 0.01)
        if self.rng.random() < rho:                       # dims also blur with overlap
            sc = p.dim_scale.get(obs_var, 1.0) * self.rng.normal(1.0, 0.01)
        if variant in ("blue", "red", "civil") and \
                self.rng.random() < self.cfg["ambiguity"].get("p_nctr_persistent_error", 0.0):
            # modified or non-standard airframe: library match is consistently wrong, on every report
            if em is not None:
                same = [c for c in EMITTERS if EMITTERS[c][0] == EMITTERS[em][0] and c not in (em, "EJ")]
                em = str(self.rng.choice(same)) if same else em
            if en is not None:
                en = str(self.rng.choice([c for c in ENGINE_CODES if c != en and c != "WB"]))
        return em, en, (p.length_m * sc, p.span_m * sc, p.tail_m * sc)

    def _transponder(self, cls, allegiance):
        c = self.cfg["iff"]
        r = self.rng.random()
        if allegiance == FRIEND:
            if cls in ("U1", "U2"):
                return "mode5_l2" if self.rng.random() < c["p_uas_iff"] else "none"
            if r > c["p_IFF_valid_friend"]:
                return "none"
            return "mode5_l2" if self.rng.random() < c["mode5_l2_fraction"] else "mode5_l1"
        if allegiance == NEUTRAL:
            if cls == "E":
                return "mode3c"
            if cls == "F":
                return "mode3c" if r < c["p_GA_squawk"] else "none"
            return "none"
        if allegiance == HOSTILE and cls not in ("U1", "U2") and \
                self.rng.random() < self.cfg["deception"]["p_IFF_spoof"]:
            return str(self.rng.choice(["spoof_mode3c", "spoof_replay"]))
        return "none"

    def _add(self, cls, variant, allegiance, role, gid, traj, target_asset=-1, **flags):
        pos, vel, alive, phase = traj
        if not alive.any():
            return None
        cls, variant = str(cls), str(variant)
        em, en, dims = self._signature(cls, variant)
        em = None if em is None else str(em)
        en = None if en is None else str(en)
        e = Entity(eid=int(self._ids.pop()), cls=cls, variant=variant, allegiance=allegiance, role=role,
                   group_id=gid, tier=PLATFORMS[cls].tier, pos=pos, vel=vel, alive=alive, phase=phase,
                   is_ooi=(cls != "K"), transponder=self._transponder(cls, allegiance) if cls != "K" else "none",
                   emitter_obs=em, engine_obs=en, dims=dims, target_asset=target_asset, flags=dict(flags))
        self.entities.append(e)
        return e

    def _members(self, leader_traj, n, cls_list, variant, allegiance, roles, gid, spacing_m, jitter_m,
                 independent_plan_fn, target_asset=-1, dalts=None):
        """Add n group members: formation followers if coordinated, else independent flights."""
        coordinated = (self.rng.random() < self.rel["coordination_strength"]) and np.isfinite(spacing_m)
        for k in range(n):
            cls = cls_list[k]
            if coordinated:
                # two-ship elements: wingman tucked close to its element lead, elements at `spacing_m`
                slot = k + 1                                   # leader is slot 0
                element, is_wing = slot // 2, slot % 2 == 1
                side = ((element + 1) // 2) * (1 if element % 2 else -1)
                off = np.array([side * spacing_m, -abs(side) * spacing_m * 0.5])
                if is_wing:
                    off = off + np.array([0.15 * spacing_m * (1 if side >= 0 else -1), -0.1 * spacing_m])
                traj = follow(*leader_traj, off, jitter_m, self.rng, dalt=(dalts[k] if dalts else 0.0))
            else:
                traj = independent_plan_fn(cls)
            self._add(cls, variant, allegiance, roles[k], gid, traj, target_asset, coordinated=coordinated)
        return coordinated

    # ------------------------------------------------------------------ Tier 2 behaviors
    def _strike_plan(self, start, target, cls, transit_alt, low_alt, speed, egress_to=None):
        d_on = self.cfg["behavior"]["descent_onset_range_km"] * 1000.0
        u = (target[:2] - start) / max(np.linalg.norm(target[:2] - start), 1.0)
        dscnt = target[:2] - u * d_on
        attack = target[:2] - u * 20_000
        egress = egress_to if egress_to is not None else start
        return [Waypoint(*dscnt, transit_alt, speed, "transit"),
                Waypoint(*attack, low_alt, speed * 1.15, "ingress"),
                Waypoint(*target[:2], low_alt, speed * 1.15, "attack"),
                Waypoint(*egress, transit_alt, speed, "egress")]

    def strike_package(self):
        T = self.T
        tgt_idx = int(self.rng.integers(len(self.world.assets)))
        target = self.world.assets[tgt_idx].pos
        start = self._threat_point(230, 300)
        t0 = int(self.rng.uniform(0, 600))
        gid = self._new_gid()
        spd = self.rng.uniform(220, 250)
        h_tr, h_lo = self.rng.uniform(6000, 9000), self.rng.uniform(100, 300)
        hdg = math.degrees(math.atan2(*(target[:2] - start)))
        plan = self._strike_plan(start, target, "B", h_tr, h_lo, spd)
        lead = fly(plan, (*start, h_tr), hdg, spd, t0, T, "B", self.rng)
        self._add("B", "red", HOSTILE, "striker", gid, lead, tgt_idx)
        n = max(int(self.cfg["composition"]["package_size"]) - 1, 0)
        n_esc = min(2 + (n > 5), n)
        n_str = n - n_esc
        cls_list = ["B"] * n_str + ["A"] * n_esc
        roles = ["striker"] * n_str + ["escort"] * n_esc
        spacing = self.rel["formation_tightness_km"] * 1000.0

        def indep(cls):
            st = self._threat_point(230, 300)
            ti = int(self.rng.integers(len(self.world.assets)))
            tg = self.world.assets[ti].pos
            p = self._strike_plan(st, tg, cls, self.rng.uniform(6000, 9000), self.rng.uniform(100, 300), spd)
            return fly(p, (*st, p[0].alt), math.degrees(math.atan2(*(tg[:2] - st))), spd,
                       int(self.rng.uniform(0, 600)), T, cls, self.rng)

        # escorts sit ahead and high when role structure is on
        dalts = [0.0] * n_str + ([1000.0] * n_esc if self.rel["package_role_structure"] else [0.0] * n_esc)
        self._members(lead[:4], n, cls_list, "red", HOSTILE, roles, gid, spacing, 60.0, indep, tgt_idx, dalts)
        # standoff jammer
        if self.rel["jammer_present"] and self.rng.random() < 0.7:
            jpos = start + (target[:2] - start) * 0.35
            orbit = [Waypoint(*(jpos + bearing_vec(hdg + 90) * 15000), 8000, 200, "orbit"),
                     Waypoint(*(jpos - bearing_vec(hdg + 90) * 15000), 8000, 200, "orbit")]
            jt = fly(orbit, (*start, 8000), hdg, 200, t0, T, "C", self.rng, loop=True)
            self._add("C", "red", HOSTILE, "jammer", gid, jt, tgt_idx, jammer=True)
        if self.rng.random() < self.cfg["deception"]["rate_airliner_mimic"]:
            self.airliner_mimic(gid, tgt_idx, t0)

    def airliner_mimic(self, gid, tgt_idx, t0):
        """A hostile striker flies an airway profile with a civil squawk, then leaves the airway and
        descends onto the package's target (spec Section 3.5)."""
        T = self.T
        target = self.world.assets[tgt_idx].pos[:2]
        threat = bearing_vec(self.world.threat_axis_deg)
        best = None
        for aw in self.world.airways:
            u = (aw.b - aw.a) / np.linalg.norm(aw.b - aw.a)
            P = aw.a + u * ((target - aw.a) @ u)               # airway point closest to the target
            d = np.linalg.norm(P - target)
            if best is None or d < best[0]:
                best = (d, P, u, aw)
        _, P, u, aw = best
        if (P - u * 250_000) @ threat < (P + u * 250_000) @ threat:
            u = -u                                             # start on the threat side
        start = P - u * 250_000
        alt = self.rng.uniform(*aw.alt_band_m)
        brk = P - u * self.cfg["behavior"]["descent_onset_range_km"] * 600.0
        plan = [Waypoint(*brk, alt, 235, "airway"),
                Waypoint(*(target + (brk - target) * 0.3), 300, 260, "ingress"),
                Waypoint(*target, 150, 260, "attack"), Waypoint(*start, alt, 235, "egress")]
        traj = fly(plan, (*start, alt), math.degrees(math.atan2(*u)), 235, max(t0 - 200, 0), T, "B", self.rng)
        e = self._add("B", "red", HOSTILE, "airliner_mimic", gid, traj, tgt_idx, airliner_mimic=True)
        if e is not None:
            e.transponder = "spoof_mode3c"

    def fighter_sweep(self):
        T = self.T
        start = self._threat_point(220, 280)
        gid = self._new_gid()
        t0 = int(self.rng.uniform(0, 400))
        spd = self.rng.uniform(280, 320)
        u = -start / np.linalg.norm(start)
        hold = start + u * (np.linalg.norm(start) - 45_000)
        plan = [Waypoint(*hold, 8000, spd, "ingress"),
                Waypoint(*(hold + bearing_vec(90 + math.degrees(math.atan2(*u))) * 20_000), 8000, 250, "orbit"),
                Waypoint(*(hold - bearing_vec(90 + math.degrees(math.atan2(*u))) * 20_000), 8000, 250, "orbit")]
        hdg = math.degrees(math.atan2(*u))
        lead = fly(plan[:1] + plan[1:], (*start, 8000), hdg, spd, t0, T, "A", self.rng)
        self._add("A", "red", HOSTILE, "sweep", gid, lead, 0)
        n = int(self.rng.integers(1, 4))

        def indep(cls):
            st = self._threat_point(220, 280)
            uu = -st / np.linalg.norm(st)
            hp = st + uu * (np.linalg.norm(st) - 45_000)
            return fly([Waypoint(*hp, 8000, spd, "ingress")], (*st, 8000), math.degrees(math.atan2(*uu)), spd,
                       int(self.rng.uniform(0, 400)), T, cls, self.rng)

        self._members(lead, n, ["A"] * n, "red", HOSTILE, ["sweep"] * n, gid,
                      self.rel["formation_tightness_km"] * 700.0, 40.0, indep, 0)

    def cap(self):
        T = self.T
        axis = self.world.threat_axis_deg
        c = bearing_vec(axis) * self.rng.uniform(60_000, 90_000)
        side = bearing_vec(axis + 90) * 20_000
        plan = [Waypoint(*(c + side), 7000, 200, "orbit"), Waypoint(*(c - side), 7000, 200, "orbit")]
        gid = self._new_gid()
        lead = fly(plan, (*(c - side), 7000), axis + 90, 200, 0, T, "A", self.rng, loop=True)
        self._add("A", "blue", FRIEND, "cap", gid, lead, -1)

        def indep(cls):
            cc = bearing_vec(axis + self.rng.uniform(-40, 40)) * self.rng.uniform(50_000, 100_000)
            p = [Waypoint(*(cc + side), 7000, 200, "orbit"), Waypoint(*(cc - side), 7000, 200, "orbit")]
            return fly(p, (*(cc - side), 7000), axis + 90, 200, 0, T, cls, self.rng, loop=True)

        self._members(lead, 1, ["A"], "blue", FRIEND, ["cap"], gid, self.rel["formation_tightness_km"] * 1000, 50, indep)

    def rtb(self):
        T = self.T
        cor = self.world.corridor
        gid = self._new_gid()
        n = int(self.rng.integers(1, 5))
        cls_lead = self.rng.choice(["A", "B"])
        t0 = int(self.rng.uniform(0, 900))
        lost = self.rng.random() < self.cfg["deception"]["rate_lost_friendly"]
        if lost:   # off-corridor route, straight in from a point rotated toward the threat axis
            st = bearing_vec(self.world.threat_axis_deg + 25) * 170_000
            plan = [Waypoint(*st * 0.5, 4000, 200, "rtb"), Waypoint(0, 0, 500, 120, "rtb")]
            start = st
        else:
            plan = [Waypoint(*cor[1], 5000, 200, "rtb"), Waypoint(*cor[2], 2000, 150, "rtb"),
                    Waypoint(*cor[3], 500, 110, "rtb")]
            start = cor[0]
        hdg = math.degrees(math.atan2(*(plan[0].x - start[0], plan[0].y - start[1])))
        lead = fly(plan, (*start, 5000), hdg, 200, t0, T, cls_lead, self.rng, end_alive_at_last=True)
        e = self._add(cls_lead, "blue", FRIEND, "rtb", gid, lead, -1, lost_friendly=lost)
        if e is not None and lost:
            e.transponder = "none"

        def indep(cls):
            return fly(plan, (*start, 5000), hdg, 200, int(self.rng.uniform(0, 900)), T, cls, self.rng,
                       end_alive_at_last=True)

        self._members(lead, n - 1, [cls_lead] * (n - 1), "blue", FRIEND, ["rtb"] * (n - 1), gid,
                      self.rel["formation_tightness_km"] * 1000, 50, indep)

    def friendly_tactical(self):
        """Friendly strike returning low and fast from the threat sector toward base. Kinematically it
        looks like a hostile ingress; IFF (when working) and track history are what separate it."""
        T = self.T
        axis = self.world.threat_axis_deg
        start = bearing_vec(axis + self.rng.uniform(-35, 35)) * self.rng.uniform(140_000, 200_000)
        base = self.world.assets[0].pos[:2]
        gid = self._new_gid()
        spd = self.rng.uniform(230, 270)
        plan = [Waypoint(*(base + (start - base) * 0.45), 600, spd, "rtb"),
                Waypoint(*(base + (start - base) * 0.08), 300, spd * 0.8, "rtb"),
                Waypoint(*base, 300, 120, "rtb")]
        t0 = int(self.rng.uniform(0, 1000))
        hdg = math.degrees(math.atan2(*(base - start)))
        cls = str(self.rng.choice(["A", "B"]))
        lead = fly(plan, (*start, 5000), hdg, spd, t0, T, cls, self.rng, end_alive_at_last=True)
        self._add(cls, "blue", FRIEND, "tactical_rtb", gid, lead)
        n = int(self.rng.integers(1, 4))

        def indep(c):
            st = bearing_vec(axis + self.rng.uniform(-35, 35)) * self.rng.uniform(140_000, 200_000)
            p = [Waypoint(*(base + (st - base) * 0.45), 600, spd, "rtb"), Waypoint(*base, 300, 120, "rtb")]
            return fly(p, (*st, 5000), math.degrees(math.atan2(*(base - st))), spd, int(self.rng.uniform(0, 1000)),
                       T, c, self.rng, end_alive_at_last=True)

        self._members(lead, n, [cls] * n, "blue", FRIEND, ["tactical_rtb"] * n, gid,
                      self.rel["formation_tightness_km"] * 1000, 50, indep)

    def airway_traffic(self):
        T = self.T
        dev = self.cfg["behavior"]["airway_deviation_sigma_km"] * 1000.0
        for _ in range(int(self.cfg["composition"]["n_airway_tracks"])):
            aw = self.world.airways[int(self.rng.integers(len(self.world.airways)))]
            a, b = (aw.a, aw.b) if self.rng.random() < 0.5 else (aw.b, aw.a)
            L = np.linalg.norm(b - a)
            u = (b - a) / L
            spd = self.rng.uniform(210, 250)
            alt = self.rng.uniform(*aw.alt_band_m)
            t0 = self.rng.uniform(-L / spd, T)              # may already be en route at t = 0
            s0 = max(0.0, -t0) * spd
            if s0 >= L:
                continue
            off = np.array([u[1], -u[0]]) * self.rng.normal(0, dev)
            start = a + u * s0 + off
            plan = [Waypoint(*(b + off), alt, spd, "airway")]
            stray = self.rng.random() < self.cfg["deception"]["rate_neutral_stray"]
            if stray:
                ast = self.world.assets[int(self.rng.integers(len(self.world.assets)))].pos[:2]
                plan = [Waypoint(*(ast + self.rng.uniform(-8000, 8000, 2)), alt * 0.6, spd * 0.8, "wander")] + plan
            traj = fly(plan, (*start, alt), math.degrees(math.atan2(*u)), spd, int(max(t0, 0)), T, "E", self.rng,
                       end_alive_at_last=True)
            self._add("E", "civil", NEUTRAL, "airway", self._new_gid(), traj, -1, neutral_stray=stray)

    def general_aviation(self):
        T = self.T
        for _ in range(int(self.cfg["composition"]["n_ga_tracks"])):
            c = self.rng.uniform(-60_000, 60_000, 2)
            wps = [Waypoint(*(c + self.rng.uniform(-20_000, 20_000, 2)), self.rng.uniform(500, 3000),
                            self.rng.uniform(55, 85), "wander") for _ in range(8)]
            traj = fly(wps, (*c, wps[0].alt), self.rng.uniform(0, 360), 70, int(self.rng.uniform(0, 600)), T, "F",
                       self.rng, loop=True)
            self._add("F", "civil", NEUTRAL, "ga", self._new_gid(), traj)

    def helo_flight(self):
        T = self.T
        hostile = self.rng.random() < 0.5
        gid = self._new_gid()
        if hostile:
            start = self._threat_point(35, 50)
            ti = int(self.rng.integers(len(self.world.assets)))
            tg = self.world.assets[ti].pos[:2]
            plan = [Waypoint(*(tg + (start - tg) * 0.3), 60, 65, "ingress"), Waypoint(*tg, 50, 60, "attack"),
                    Waypoint(*start, 80, 65, "egress")]
        else:
            ti = -1
            a0, a1 = self.world.assets[0].pos[:2], self.world.assets[-1].pos[:2]
            start = a0 + self.rng.uniform(-3000, 3000, 2)
            plan = [Waypoint(*a1, 150, 55, "transit"), Waypoint(*a0, 150, 55, "transit")]
        t0 = int(self.rng.uniform(0, 900))
        hdg = math.degrees(math.atan2(plan[0].x - start[0], plan[0].y - start[1]))
        lead = fly(plan, (*start, 100), hdg, 60, t0, T, "R", self.rng, loop=not hostile)
        var, alg = ("red", HOSTILE) if hostile else ("blue", FRIEND)
        self._add("R", var, alg, "helo", gid, lead, ti)

        def indep(cls):
            st = start + self.rng.uniform(-5000, 5000, 2)
            return fly(plan, (*st, 100), hdg, 60, int(self.rng.uniform(0, 900)), T, cls, self.rng, loop=not hostile)

        self._members(lead, 1, ["R"], var, alg, ["helo"], gid, self.rel["formation_tightness_km"] * 300, 20, indep, ti)

    def orbiter(self, cls, variant, allegiance, role, centre, alt, spd, half_len=15_000):
        T = self.T
        axis = self.world.threat_axis_deg
        side = bearing_vec(axis + 90) * half_len
        plan = [Waypoint(*(centre + side), alt, spd, "orbit"), Waypoint(*(centre - side), alt, spd, "orbit")]
        traj = fly(plan, (*(centre - side), alt), axis + 90, spd, 0, T, cls, self.rng, loop=True)
        self._add(cls, variant, allegiance, role, self._new_gid(), traj)

    # ------------------------------------------------------------------ Tier 1 behaviors
    def _drone_attack_plan(self, start, tgt, cls, spd):
        alt = self.rng.uniform(40, 250)
        mid = start + (tgt - start) * 0.5 + self.rng.uniform(-800, 800, 2)
        return [Waypoint(*mid, alt, spd, "ingress"), Waypoint(*tgt, alt * 0.5, spd, "attack")]

    def drone_swarm(self):
        T = self.T
        ti = int(self.rng.integers(len(self.world.assets)))
        tgt = self.world.assets[ti].pos[:2]
        start = self._threat_point(6, 10, about=tgt)
        n = int(self.cfg["composition"]["swarm_size"])
        cls = self.rng.choice(["U1", "U2"], p=[0.6, 0.4])
        spd = self.rng.uniform(15, 28) if cls == "U1" else self.rng.uniform(25, 40)
        t0 = int(self.rng.uniform(200, 1100))
        gid = self._new_gid()
        cohesion = float(self.rel["swarm_cohesion"])
        autonomous = self.rng.random() < self.cfg["composition"]["p_drone_autonomous"]
        bird_like = self.rng.random() < self.cfg["deception"]["rate_bird_like_drone"]
        if bird_like:   # meander toward the target at bird speed and height, like a flock would
            spd = self.rng.uniform(9, 15)
            plan = [Waypoint(*(start + (tgt - start) * f + self.rng.uniform(-700, 700, 2)), self.rng.uniform(30, 200),
                             spd, "ingress") for f in np.linspace(0.12, 0.9, 8)] + \
                   [Waypoint(*tgt, 40, spd, "attack")]
        else:
            plan = self._drone_attack_plan(start, tgt, cls, spd)
        hdg = math.degrees(math.atan2(*(tgt - start)))
        lead = fly(plan, (*start, plan[0].alt), hdg, spd, t0, T, cls, self.rng, end_alive_at_last=True)
        self._add(cls, "red", HOSTILE, "swarm", gid, lead, ti, autonomous=autonomous, bird_like=bird_like)
        for k in range(n - 1):
            if cohesion > 0 and self.rng.random() < max(cohesion, 0.05):
                off = self.rng.uniform(-60, 60, 2) / max(cohesion, 0.1)
                traj = follow(*lead, off, 8.0 + 60.0 * (1 - cohesion), self.rng, dalt=self.rng.uniform(-20, 20))
            else:   # independent drone: own spawn, own target, own timing
                tj = int(self.rng.integers(len(self.world.assets)))
                tg = self.world.assets[tj].pos[:2]
                st = self._threat_point(6, 10, about=tg)
                p = self._drone_attack_plan(st, tg, cls, spd)
                traj = fly(p, (*st, p[0].alt), self.rng.uniform(0, 360), spd, int(self.rng.uniform(200, 1100)), T,
                           cls, self.rng, end_alive_at_last=True)
            self._add(cls, "red", HOSTILE, "swarm", gid, traj, ti, autonomous=autonomous, bird_like=bird_like)

    def own_uas(self):
        T = self.T
        for _ in range(int(self.cfg["composition"]["n_own_uas"])):
            a = self.world.assets[int(self.rng.integers(len(self.world.assets)))].pos[:2]
            cls = self.rng.choice(["U1", "U2"])
            wps = [Waypoint(*(a + self.rng.uniform(-2000, 2000, 2)), self.rng.uniform(60, 200), 12, "loiter",
                            hold_s=float(self.rng.uniform(0, 30))) for _ in range(6)]
            traj = fly(wps, (*a, 80), 0, 10, int(self.rng.uniform(0, 600)), T, cls, self.rng, loop=True)
            self._add(cls, "blue", FRIEND, "own_uas", self._new_gid(), traj, -1, operating_area=a.tolist())

    def hobbyist(self):
        T = self.T
        for _ in range(int(self.cfg["composition"]["n_hobbyist"])):
            c = self.world.town + self.rng.uniform(-1500, 1500, 2)
            wps = [Waypoint(*(c + self.rng.uniform(-600, 600, 2)), self.rng.uniform(20, 110), self.rng.uniform(4, 14),
                            "wander", hold_s=float(self.rng.uniform(0, 40))) for _ in range(10)]
            traj = fly(wps, (*c, 30), 0, 5, int(self.rng.uniform(0, 1500)), min(T, int(self.rng.uniform(900, 1800)) + 600),
                       "U1", self.rng)
            pos, vel, alive, ph = traj
            pad = lambda x, fill: np.concatenate([x, np.full((T - len(x),) + x.shape[1:], fill, x.dtype)])
            traj = (pad(pos, np.nan), pad(vel, np.nan), pad(alive, False), pad(ph, -1))
            self._add("U1", "civil", NEUTRAL, "hobbyist", self._new_gid(), traj)

    def bird_flock(self):
        T = self.T
        lo, hi = self.cfg["composition"]["bird_flock_size"]
        n = int(self.rng.integers(lo, hi + 1))
        c = self.rng.uniform(-6000, 6000, 2)
        wps = [Waypoint(*(c + self.rng.uniform(-3000, 3000, 2)), self.rng.uniform(30, 300), self.rng.uniform(8, 15),
                        "wander") for _ in range(12)]
        gid = self._new_gid()
        lead = fly(wps, (*c, 80), self.rng.uniform(0, 360), 10, 0, T, "K", self.rng, loop=True)
        self._add("K", "none", NONE, "bird", gid, lead)
        for _ in range(n - 1):
            traj = follow(*lead, self.rng.uniform(-40, 40, 2), 25.0, self.rng, dalt=self.rng.uniform(-15, 15))
            self._add("K", "none", NONE, "bird", gid, traj)

    # ------------------------------------------------------------------ IFF outages
    def _iff_outages(self):
        c = self.cfg["iff"]
        frac, mean_down = c.get("outage_fraction", 0.0), c.get("outage_mean_s", 180.0)
        if frac <= 0:
            return
        p_end = 1.0 / mean_down
        civil_scale = c.get("civil_outage_scale", 0.1)
        for e in self.entities:
            if e.transponder not in ("mode5_l1", "mode5_l2", "mode3c"):
                continue
            fe = frac if e.transponder != "mode3c" else frac * civil_scale     # civil transponders rarely fail
            p_start = fe / max(1 - fe, 1e-6) * p_end
            up = np.ones(self.T, bool)
            state = self.rng.random() >= fe
            draws = self.rng.random(self.T)
            for t in range(self.T):
                if state and draws[t] < p_start:
                    state = False
                elif not state and draws[t] < p_end:
                    state = True
                up[t] = state
            e.iff_up = up

    # ------------------------------------------------------------------ emitters
    def _emitter_timelines(self):
        emcon = self.cfg["deception"]["emcon_switch_on_range_km"] * 1000.0
        assets = np.stack([a.pos[:2] for a in self.world.assets])
        for e in self.entities:
            on = np.zeros(self.T, bool)
            if e.emitter_obs is None and not e.flags.get("jammer"):
                e.emitting = on
                continue
            al = e.alive
            if e.flags.get("jammer"):
                # jams while any member of its package is within 150 km of the target
                mates = [m for m in self.entities if m.group_id == e.group_id and m is not e]
                d = np.full(self.T, np.inf)
                for m in mates:
                    dm = np.linalg.norm(m.pos[:, :2] - assets[max(e.target_asset, 0)], axis=1)
                    d = np.fmin(d, np.where(m.alive, dm, np.inf))
                on = al & (d < 150_000)
                e.emitter_obs = "EJ"
            elif e.allegiance == HOSTILE and e.cls in ("A", "B", "R"):
                dmin = np.min(np.linalg.norm(e.pos[:, None, :2] - assets[None], axis=2), axis=1)
                on = al & (dmin < emcon)
            elif e.allegiance == HOSTILE and e.cls in ("U1", "U2"):
                on = al & (not e.flags.get("autonomous", False))
            elif e.cls in ("C", "E", "U1", "U2"):
                on = al.copy()
            else:
                on = al & (self.rng.random() < 0.6)
            e.emitting = on

    # ------------------------------------------------------------------ build
    def build(self) -> list[Entity]:
        c = self.cfg["composition"]
        axis = self.world.threat_axis_deg
        if 2 in self.tiers:
            for _ in range(int(c["n_hostile_groups"])):
                self.strike_package()
            for _ in range(int(c["n_fighter_sweeps"])):
                self.fighter_sweep()
            for _ in range(int(c["n_cap"])):
                self.cap()
            for _ in range(int(c["n_rtb_groups"])):
                self.rtb()
            for _ in range(int(c.get("n_friendly_tactical", 0))):
                self.friendly_tactical()
            self.airway_traffic()
            self.general_aviation()
            for _ in range(int(c["n_helo_flights"])):
                self.helo_flight()
            for _ in range(int(c["n_hostile_uas_g3"])):
                self.orbiter("D", "red", HOSTILE, "recon", bearing_vec(axis + self.rng.uniform(-30, 30)) * 90_000,
                             5000, 60, 10_000)
            for _ in range(int(c["n_friendly_aew"])):
                self.orbiter("C", "blue", FRIEND, "aew", bearing_vec(axis + 180) * 80_000, 9000, 210)
            for _ in range(int(c["n_hostile_aew"])):
                self.orbiter("C", "red", HOSTILE, "aew", bearing_vec(axis) * 220_000, 9000, 210)
        if 1 in self.tiers:
            for _ in range(int(c["n_drone_swarms"])):
                self.drone_swarm()
            self.own_uas()
            self.hobbyist()
            for _ in range(int(c["bird_flocks"])):
                self.bird_flock()
        self._emitter_timelines()
        self._iff_outages()
        # hypersonic (Tier 3) profile tracks: added last, on their own random stream
        from .hypersonic_profiles import add_profile_tracks
        self.tier3_segments = add_profile_tracks(self)
        return self.entities


def build_population(cfg, world, rng, return_tier3=False):
    pop = Population(cfg, world, rng)
    ents = pop.build()
    return (ents, pop.tier3_segments) if return_tier3 else ents


PHASE_NAMES = {v: k for k, v in PHASES.items()}
