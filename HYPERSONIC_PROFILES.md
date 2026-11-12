# Hypersonic (Tier 3) profile tracks

`cidgen/hypersonic_profiles.py` plays back track shapes that you describe leg by leg in a scenario
YAML. It contains no vehicle model: no aerodynamics, propulsion, energy or guidance, and no vehicle
values. Every number comes from your scenario file. The resulting tracks go through the same sensors,
tracker, features, labels, CID models and ThreatPrioritizer as every other entity.

```
python -m cidgen.generate --scenario S12 --seed 1 --out runs     # S0 + the example profiles
python tests/test_hypersonic_profiles.py                         # about 1 minute, CPU only
```

## What a segment is

A segment is one leg of flight, described by how it starts and how it ends. Worked example (an
airliner arrival, also in `S12_hypersonic.yaml`):

```yaml
AIRLINER-ARRIVAL:
  class: E
  allegiance: neutral
  variant: civil
  segments:
    - {phase: cruise,   duration_s: 600, alt_km: [11.0, 11.0], speed_mps: [230, 230]}
    - {phase: descent,  duration_s: 900, alt_km: [11.0, 3.0],  speed_mps: [230, 150]}
    - {phase: approach, duration_s: 300, alt_km: [3.0, 0.5],   speed_mps: [150, 80],
       weave_g: 0.2, weave_period_s: 40}
```

| Field | Meaning |
| --- | --- |
| `phase` | Label written to `truth_states.phase` and `labels.intent_phase`; new labels are registered automatically |
| `duration_s` | Length of the leg; legs run back to back |
| `alt_km: [start, end]` | Altitude, linear across the leg; climb or descent rate follows from it |
| `speed_mps: [start, end]` | Total speed, linear across the leg; horizontal speed is what remains after the vertical rate |
| `weave_g`, `weave_period_s` | Optional. Constant lateral acceleration whose direction flips about every `weave_period_s` (randomized x0.5 to x1.5). Turn rate = acceleration / speed |

At t = 750 s the airliner is 150 s (1/6) into the descent leg: altitude 11.0 - 8.0/6 = 9.67 km,
speed 230 - 80/6 = 216.7 m/s, phase `descent`. The test checks exactly these numbers.

Any number may be a distribution, so one profile gives a different track every time:

| Spec | Parameters |
| --- | --- |
| `{dist: uniform, low, high}` | |
| `{dist: normal, mean, std}` | |
| `{dist: truncnormal, mean, std, low, high}` | |
| `{dist: lognormal, median, sigma_ln}` | |
| `{dist: triangular, low, mode, high}` | |
| `{dist: choice, options, weights}` | weights optional |

A start value of `null` carries over the previous leg's end value, which keeps legs continuous when
that end value was drawn from a distribution.

## Profile fields

| Field | Default | Notes |
| --- | --- | --- |
| `class` | `H` | Any `library.PLATFORMS` code. `H` is the generic Tier 3 placeholder |
| `allegiance` | `hostile` | `friend`, `neutral` or `hostile` |
| `variant` | `red` | Library variant key (sets emitter / engine codes for Tier 1-2 classes) |
| `role` | profile name | Written to the entity and label tables |
| `rcs_dbsm` | class median | RCS level for this track (number or distribution) |
| `rcs_mod_db` | none | `[[t_s, dB], ...]` added to the RCS over profile time, linearly interpolated. Use it for any time-varying signature assumption your team adopts |

## Placement and timing

You never give positions. For each track the generator picks an asset, an approach bearing (threat
axis +/- `ingress_halfwidth_deg`) and an arrival time in `arrival_window_s`, then rotates and places the
whole path so that it ends on the asset at that time. A track that starts before t = 0 is already in
flight when the run opens. The distance covered follows from your speeds and durations.

## Raid waves

`raid:` can be a simple mapping, `{PROFILE: count}`, where every track shares the block-level
`arrival_window_s`. It can also be a list of waves, played in order, each timed one of two ways:

```yaml
raid:
  - {profile: HGV-GENERIC, count: 4, arrival_window_s: [600, 1100]}   # absolute arrival window
  - {profile: J,           count: 6, start_after_prev_s: [50, 70]}    # starts 50-70 s after the previous
                                                                      # wave's last arrival
```

| Key | Meaning |
| --- | --- |
| `profile`, `count` | Which profile and how many tracks |
| `arrival_window_s: [lo, hi]` | Each track arrives on its asset at a random time in this absolute window |
| `start_after_prev_s: [lo, hi]` | Each track **starts** its profile this long after the previous wave's last arrival; it arrives one profile-duration later |
| `ingress_halfwidth_deg` | Optional per-wave override of the approach spread |

Give each wave exactly one of the two timing keys; the first wave must use `arrival_window_s`. Runs are
1800 s long by default, and a track that would arrive after the end of the run is cut off, so leave
room for later waves. The wave number is written to `hypersonic_segments.wave` and `entities.flag_wave`.

## Scenario block

```yaml
hypersonic:
  profiles: {NAME: {...}, ...}
  raid: {NAME: count, ...}                   # or a list of waves (above)
  arrival_window_s: [600, 1700]
  ingress_halfwidth_deg: 30
  track_coast_s: 6.0                          # Tier 3 track drop time (or tracking.tiers.3.coast_s)
  tracker: {q: 1000.0, vel_sigma_mps: 2000.0} # Tier 3 filter (or tracking.tiers.3; see below)
```

## What changed elsewhere (all marked "hypersonic (Tier 3)" in the code)

| File | Change |
| --- | --- |
| `library.py` | Class `H`, Tier 3 generic placeholder (signature values are not estimates); `TIER3` list |
| `entities.py` | `Entity.rcs_mult`; `Population.build()` adds profile tracks last, on their own random stream |
| `sensors/common.py`, `sensors/radar.py` | Optional per-entity, per-second RCS multiplier |
| `tracking.py` | Per-tier tracker settings (`tracking.tiers`) and the `tracking.tier_source` option |
| `generate.py` | Chooses each track's filter tier; writes `hypersonic_segments` (and `track_filter_tier` in observed mode) |
| `models/bn.py` | Tier 3 BN (speed, altitude, climb, RCS, IFF, closing, sector); `observable_tier` returns 3 for speed >= 1000 m/s |
| `configs/threat/doctrine_default.yaml` | `tier3_high_speed` lethality row (notional value 1.0) |
| `configs/scenarios/S12_hypersonic.yaml` | Example scenario |

Scenarios without a `hypersonic.raid` block are unchanged: S0 regenerates byte-for-byte, and in S12
every non-profile entity and truth row is identical to S0 for the same seed.

## Tracker settings by tier

The Kalman filter's settings can be set per tier in any scenario YAML (top level, beside `run:`):

```yaml
tracking:
  tier_source: truth        # truth (default) or observed
  tiers:
    1: {q: 3.0,    vel_sigma_mps: [15, 15, 5],    coast_s: 6.0}
    2: {q: 30.0,   vel_sigma_mps: [100, 100, 30], coast_s: 12.0}
    3: {q: 1000.0, vel_sigma_mps: 2000.0,         coast_s: 6.0}
  acquisition: {window_s: 10.0}   # observed mode only
```

The values shown are the defaults, so give only what you change. `q` is the white-acceleration
spectral density (m^2/s^3) in Q = q [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]]; `vel_sigma_mps` is the initial
velocity sigma (one value or [e, n, u]); `coast_s` is how long a track survives without an update.
The older `hypersonic.tracker` and `hypersonic.track_coast_s` keys still work and set Tier 3; `tracking.tiers`
overrides them.

`tier_source` decides which tier's settings each track gets:

| Value | Tier used for the filter |
| --- | --- |
| `truth` | The entity's true tier (v0.6.1 behavior; existing runs are reproduced exactly) |
| `observed` | Judged from the track's first radar reports over `acquisition.window_s`: speed from the displacement across the window, median altitude and median measured RCS, with the same rules as `models.bn.observable_tier` (speed >= 1000 m/s gives Tier 3; slow, low and small gives Tier 1). No truth is used. The choice is written to `track_filter_tier` for diagnosis |

`observed` changes tracking for every track, so runs made with it are not comparable to runs made
with `truth`. Use one setting for every scenario you compare, and regenerate them all if you switch.

## Output

`hypersonic_segments` (one row per track per leg) records every drawn value: profile, phase, start
and end times, altitudes, speeds, weave, target asset, approach bearing and arrival time. It is a
**label** table: never use it as a model input.

## Things to know

- `observable_tier` uses speed >= 1000 m/s (`TIER3_MIN_SPEED_MPS` in `models/bn.py`). A profile slower
  than that is routed to the Tier 2 BN, and a Tier 3 track can read as Tier 2 for its first few
  updates while its velocity estimate converges.
- The Tier 3 BN is fitted like the others. With no Tier 3 rows in training it falls back to its prior.
- The tracker is a constant-velocity filter. Weaving fast tracks will lag; `q` trades lag for noise (see Tracker settings by tier).
- In the default doctrine, the `imminent` rule floors every non-friend under 60 s at the same score
  (0.6), so a fast Tier 3 track and a drone already at an asset can tie. Whether Tier 3 should break
  that tie is a doctrine decision for your team.
- The two profiles in S12 only show the format and exercise the code. `PLUMBING-TEST` is an artificial
  straight-line target, not a model of anything.
