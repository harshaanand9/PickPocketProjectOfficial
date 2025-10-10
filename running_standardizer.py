# running_standardizer.py
from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class _Welford:
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # sum of squared diffs

    def observe(self, x: float) -> None:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def mean_sd(self) -> tuple[float, float]:
        if self.n < 2:
            return (self.mean, 0.0)
        var = self.M2 / (self.n - 1)
        return (self.mean, math.sqrt(var) if var > 0 else 0.0)

class SeasonRunningZ:
    """
    Running z-score standardizer keyed by (season, feature).
    Usage pattern per season, per date (strict 'day-before' semantics):
      - call begin_date(date_str)
      - for each game that day:
            z = std.standardize(season, feature, value)
            std.stage(season, feature, value)   # stage today's obs
      - call end_date()  -> commits today's staged obs into running stats
    """
    def __init__(self):
        self._stats = defaultdict(lambda: defaultdict(_Welford))      # season -> feature -> Welford
        self._staged = defaultdict(lambda: defaultdict(list))         # season -> feature -> [values observed today]
        self._baselines = {}  # season -> feature -> (mean, sd) from previous season (set by caller)
        self._current_date = None

    # ---------- date gating ----------
    def begin_date(self, date_str: str) -> None:
        # Start a new processing day; ensures 'day-before' only
        self._current_date = date_str
        # (no commit here; commit happens at end_date)

    def end_date(self) -> None:
        # Commit all staged values to running stats
        for season, fmap in self._staged.items():
            for feat, vals in fmap.items():
                wf = self._stats[season][feat]
                for v in vals:
                    wf.observe(v)
        self._staged.clear()
        self._current_date = None

    # ---------- prev-season fallback ----------
    def set_prev_season_baseline(self, season: str, baseline_map: dict[str, tuple[float, float]]) -> None:
        """Provide previous season final (mean, sd) per feature for this season."""
        self._baselines[season] = dict(baseline_map or {})

    # ---------- main API ----------
    def stage(self, season: str, feature: str, value: float) -> None:
        """Stage today's observation; it will count only after end_date()."""
        self._staged[season][feature].append(value)

    def standardize(self, season: str, feature: str, value: float) -> float:
        """Return z = (value - mean_so_far) / sd_so_far with first-game fallback."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return math.nan

        mean, sd = self._stats[season][feature].mean_sd()
        if self._stats[season][feature].n < 2:
            # try prev-season baseline if present
            base = self._baselines.get(season, {}).get(feature)
            if base is not None:
                mean, sd = base
        if sd is None or sd == 0.0 or math.isnan(sd):
            return 0.0  # safe default: centered but no spread info yet
        return (value - mean) / sd

