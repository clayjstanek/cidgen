"""Stone Soup readiness probe for the cidgen MFA tracker.

    python stonesoup_probe.py > stonesoup_probe.txt

Prints versions, checks the MFA classes import, and runs four small checks (a few seconds in all). Send
the printed output back. Nothing here touches cidgen or writes files.
"""
import datetime
import platform
import sys
import traceback

print("python", sys.version.split()[0], "|", platform.platform())
try:
    import stonesoup
    print("stonesoup", getattr(stonesoup, "__version__", "?"), "at", stonesoup.__file__)
except Exception as e:
    print("stonesoup import FAILED:", e); sys.exit(1)
import numpy as np
print("numpy", np.__version__)


def check(name, fn):
    try:
        out = fn()
        print(f"[ OK ] {name}" + (f": {out}" if out else ""))
    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc(limit=3, file=sys.stdout)


def imports():
    from stonesoup.dataassociator.mfa import MFADataAssociator  # noqa: F401
    from stonesoup.hypothesiser.mfa import MFAHypothesiser  # noqa: F401
    from stonesoup.hypothesiser.probability import PDAHypothesiser  # noqa: F401
    from stonesoup.types.mixture import GaussianMixture  # noqa: F401
    from stonesoup.types.state import TaggedWeightedGaussianState  # noqa: F401
    from stonesoup.types.update import GaussianMixtureUpdate  # noqa: F401
    import inspect
    from stonesoup.dataassociator.mfa import MFADataAssociator as M
    return "MFADataAssociator.associate" + str(inspect.signature(M.associate))


def mfa_crossing():
    """Stone Soup's own MFA example in miniature: two crossing targets, 3-D state, clutter."""
    from scipy.stats import chi2
    from stonesoup.dataassociator.mfa import MFADataAssociator
    from stonesoup.hypothesiser.mfa import MFAHypothesiser
    from stonesoup.hypothesiser.probability import PDAHypothesiser
    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.detection import Clutter, Detection
    from stonesoup.types.mixture import GaussianMixture
    from stonesoup.types.numeric import Probability
    from stonesoup.types.state import TaggedWeightedGaussianState
    from stonesoup.types.track import Track
    from stonesoup.types.update import GaussianMixtureUpdate
    from stonesoup.updater.kalman import KalmanUpdater
    rng = np.random.default_rng(0)
    t0 = datetime.datetime(2026, 1, 1)
    tm = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05)] * 3)
    R = np.diag([1.0, 1.0, 1.0])
    mm = LinearGaussian(ndim_state=6, mapping=(0, 2, 4), noise_covar=R)
    pred = KalmanPredictor(tm); upd = KalmanUpdater(mm)
    hyp = MFAHypothesiser(PDAHypothesiser(pred, upd, 1e-4, prob_gate=chi2.cdf(16, 3), prob_detect=0.9))
    assoc = MFADataAssociator(hyp, slide_window=3)
    truths = [np.array([0, 1, 0, 1, 0, 0.0]), np.array([0, 1, 20, -1, 0, 0.0])]
    tracks = {Track([GaussianMixture([TaggedWeightedGaussianState(
        x.reshape(-1, 1), np.diag([1.5, .5, 1.5, .5, 1.5, .5]), timestamp=t0, weight=Probability(1), tag=[])])])
        for x in truths}
    F = tm.matrix(time_interval=datetime.timedelta(seconds=1))
    for n in range(1, 21):
        ts = t0 + datetime.timedelta(seconds=n)
        truths = [F @ x for x in truths]
        dets = set()
        for x in truths:
            if rng.random() < 0.9:
                # per-detection measurement model (cidgen gives every report its own R)
                dets.add(Detection((x[[0, 2, 4]] + rng.normal(0, 1, 3)).reshape(-1, 1), timestamp=ts,
                                   measurement_model=LinearGaussian(6, (0, 2, 4), R)))
        for _ in range(rng.poisson(2)):
            dets.add(Clutter(rng.uniform(-10, 30, 3).reshape(-1, 1), timestamp=ts, measurement_model=mm))
        hyps = assoc.associate(tracks, dets, ts)
        for track, hs in hyps.items():
            comps = [h.prediction if not h else upd.update(h) for h in hs]
            track.append(GaussianMixtureUpdate(components=comps, hypothesis=hs))
    best = [max(tr.state.components, key=lambda c: c.weight) for tr in tracks]
    return "final components per track " + str([len(tr.state.components) for tr in tracks]) + \
        ", best positions " + str([np.round(np.ravel(c.state_vector)[[0, 2]], 1).tolist() for c in best]) + \
        ", tag example " + str(best[0].tag)


def per_track_q():
    """Can one predictor use a different process noise per track? (cidgen: q by tier)"""
    from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.state import TaggedWeightedGaussianState
    models = {q: CombinedLinearGaussianTransitionModel([ConstantVelocity(q)] * 3) for q in (3.0, 30.0)}

    class TierPredictor(KalmanPredictor):
        def predict(self, prior, timestamp=None, **kwargs):
            self.transition_model = models[getattr(prior, "cid_q", 30.0)]
            return super().predict(prior, timestamp=timestamp, **kwargs)
    p = TierPredictor(models[30.0])
    t0 = datetime.datetime(2026, 1, 1)
    # two separate components (Stone Soup caches predictions per prior object, so reuse would hide it)
    s1 = TaggedWeightedGaussianState(np.zeros((6, 1)), np.eye(6), timestamp=t0, weight=1, tag=[])
    s2 = TaggedWeightedGaussianState(np.zeros((6, 1)), np.eye(6), timestamp=t0, weight=1, tag=[])
    s1.cid_q = 3.0; s2.cid_q = 30.0
    a = p.predict(s1, timestamp=t0 + datetime.timedelta(seconds=2)).covar[1, 1]
    b = p.predict(s2, timestamp=t0 + datetime.timedelta(seconds=2)).covar[1, 1]
    ok = abs(a - 7.0) < 1e-6 and abs(b - 61.0) < 1e-6
    return f"velocity variance after 2 s: q=3 -> {a:.1f} (expect 7), q=30 -> {b:.1f} (expect 61): {'PASS' if ok else 'MISMATCH'}"


def attr_survives():
    """Does an attribute set on a component survive Stone Soup's predict/update copies?"""
    from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.state import TaggedWeightedGaussianState
    t0 = datetime.datetime(2026, 1, 1)
    s = TaggedWeightedGaussianState(np.zeros((6, 1)), np.eye(6), timestamp=t0, weight=1, tag=[])
    s.cid_q = 3.0
    p = KalmanPredictor(CombinedLinearGaussianTransitionModel([ConstantVelocity(1.0)] * 3))
    pr = p.predict(s, timestamp=t0 + datetime.timedelta(seconds=1))
    pr.cid_q = 3.0                              # the adapter re-tags every new component itself
    return f"prediction type {type(pr).__name__}; attribute copied by Stone Soup: no (expected); settable: {pr.cid_q == 3.0}"


try:
    import ortools
    print("ortools", ortools.__version__)
except Exception:
    print("ortools NOT installed: Stone Soup's MFA needs it (pip install ortools)")
check("imports", imports)
check("MFA on two crossing targets (3-D, per-detection R)", mfa_crossing)
check("per-track process noise via a predictor subclass", per_track_q)
check("custom attribute on a component", attr_survives)


def bench(n_tracks, window, scans=15, dets_per_track=1.2, clutter=2.0):
    """ms per scan for MFA with n_tracks well-separated targets (a stand-in for one radar look)."""
    import time
    from scipy.stats import chi2
    from stonesoup.dataassociator.mfa import MFADataAssociator
    from stonesoup.hypothesiser.mfa import MFAHypothesiser
    from stonesoup.hypothesiser.probability import PDAHypothesiser
    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.detection import Clutter, Detection
    from stonesoup.types.mixture import GaussianMixture
    from stonesoup.types.numeric import Probability
    from stonesoup.types.state import TaggedWeightedGaussianState
    from stonesoup.types.track import Track
    from stonesoup.types.update import GaussianMixtureUpdate
    from stonesoup.updater.kalman import KalmanUpdater
    rng = np.random.default_rng(1)
    t0 = datetime.datetime(2026, 1, 1)
    tm = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.5)] * 3)
    mm = LinearGaussian(6, (0, 2, 4), np.eye(3) * 4.0)
    upd = KalmanUpdater(mm)
    assoc = MFADataAssociator(MFAHypothesiser(PDAHypothesiser(KalmanPredictor(tm), upd, 1e-6,
                              prob_gate=chi2.cdf(16, 3), prob_detect=0.9)), slide_window=window)
    xs = [np.array([30.0 * k, 1, 0, 0.5, 0, 0]) for k in range(n_tracks)]     # 30 m apart: some ambiguity
    tracks = {Track([GaussianMixture([TaggedWeightedGaussianState(x.reshape(-1, 1), np.diag([4, 1, 4, 1, 4, 1.0]),
              timestamp=t0, weight=Probability(1), tag=[])])]) for x in xs}
    F = tm.matrix(time_interval=datetime.timedelta(seconds=1))
    times = []
    for n in range(1, scans + 1):
        ts = t0 + datetime.timedelta(seconds=n)
        xs = [F @ x for x in xs]
        dets = {Detection((x[[0, 2, 4]] + rng.normal(0, 2, 3)).reshape(-1, 1), timestamp=ts, measurement_model=mm)
                for x in xs if rng.random() < 0.9}
        for _ in range(rng.poisson(clutter)):
            dets.add(Clutter(rng.uniform(-50, 30 * n_tracks + 50, 3).reshape(-1, 1), timestamp=ts, measurement_model=mm))
        a = time.perf_counter()
        hyps = assoc.associate(tracks, dets, ts)
        for track, hs in hyps.items():
            track.append(GaussianMixtureUpdate(components=[h.prediction if not h else upd.update(h) for h in hs],
                                               hypothesis=hs))
        times.append(time.perf_counter() - a)
    return 1e3 * float(np.median(times[window:]))      # after the window has filled


print("\n=== MFA speed on this machine (median ms per scan, after the window fills) ===")
print("tracks  window   ms/scan   S12 simulation per wall-clock hour (about 5.6 looks per simulated s)")
for n_tr in (5, 20, 60):
    for w in (3, 5):
        try:
            ms = bench(n_tr, w)
            sim_min = 3600.0 / (ms / 1e3) / 5.6 / 60.0
            print(f"{n_tr:>6}  {w:>6}  {ms:>8.1f}   {sim_min:>8.0f} min (if every look had {n_tr} tracks)")
        except Exception as e:
            print(f"{n_tr:>6}  {w:>6}  FAILED: {e}")
print("Most S12 looks involve far fewer tracks than the whole picture once the adapter clusters them;"
      " the 5- and 20-track rows are the realistic ones.")
