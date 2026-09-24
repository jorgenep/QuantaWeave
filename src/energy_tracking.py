"""GPU electricity-cost tracking for a training run.

Reads the driver's hwmon energy counter (Intel `xe` exposes one as `energy1_input`/`energy1_label
== "card"`; other drivers that expose the same convention work too) and turns the running delta
into cumulative kWh and an estimated dollar cost, written into metrics.jsonl alongside loss/lr/etc.

Best-effort by design: a host with no such sysfs entry (a different GPU vendor, a container without
sysfs mounted, CPU-only training) just gets `None` back from every field rather than a crash --
this is a nice-to-have observability feature, not something training should ever depend on.

Cumulative totals survive `--resume`: at construction, if the target metrics.jsonl already has
rows (from an earlier, interrupted run of this study), the last logged cumulative kWh is read back
and continued rather than restarted from zero.
"""

import glob
import json
import time
from pathlib import Path
from typing import Optional

JOULES_PER_KWH = 3.6e6


def _read_card_energy_uj() -> Optional[int]:
    """Cumulative microjoules from the GPU's "card" energy sensor, or None if not exposed."""
    for hwmon in glob.glob("/sys/class/hwmon/hwmon*") + glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*"):
        for energy_input in glob.glob(f"{hwmon}/energy*_input"):
            label_path = energy_input.replace("_input", "_label")
            try:
                if Path(label_path).read_text().strip() != "card":
                    continue
                return int(Path(energy_input).read_text().strip())
            except OSError:
                continue
    return None


def _last_logged_kwh(metrics_path: Path) -> float:
    try:
        with metrics_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = min(size, 8192)
            f.seek(size - block)
            tail = f.read().splitlines()
        for line in reversed(tail):
            if not line.strip():
                continue
            record = json.loads(line)
            if "energy_kwh_cumulative" in record:
                return float(record["energy_kwh_cumulative"])
            return 0.0
    except (OSError, ValueError):
        pass
    return 0.0


class PowerCostTracker:
    """Call ``sample()`` each time you log metrics; merge the returned dict into that record."""

    def __init__(self, rate_usd_per_kwh: Optional[float], resume_from: Optional[Path] = None) -> None:
        self.rate = rate_usd_per_kwh
        self.energy_kwh = _last_logged_kwh(resume_from) if resume_from and resume_from.exists() else 0.0
        self._last_uj = _read_card_energy_uj()
        self._last_time = time.perf_counter()
        self.available = self._last_uj is not None

    def sample(self) -> dict:
        if not self.available:
            return {}
        now = time.perf_counter()
        cur_uj = _read_card_energy_uj()
        dt = now - self._last_time
        self._last_time = now
        power_watts = None
        if cur_uj is not None and self._last_uj is not None and dt > 0:
            delta_uj = cur_uj - self._last_uj
            if delta_uj >= 0:  # a negative delta means the counter wrapped; skip that interval
                power_watts = (delta_uj / 1e6) / dt
                self.energy_kwh += delta_uj / 1e6 / JOULES_PER_KWH
        if cur_uj is not None:
            self._last_uj = cur_uj
        record = {"gpu_power_watts": power_watts, "energy_kwh_cumulative": self.energy_kwh}
        if self.rate is not None:
            record["cost_usd_cumulative"] = self.energy_kwh * self.rate
        return record
