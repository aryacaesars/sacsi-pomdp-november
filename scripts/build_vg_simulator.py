"""Refresh the offline VG1/VG2 simulator from this release's inference inputs."""
import base64
import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "00_Dataset" / "Processed"
PAGE = ROOT / "Dashboard" / "assets" / "Simulator_SACSI_VG1_VG2.html"
CHECKPOINT = ROOT / "Checkpoints/SACSI_Full/sacsi_full_seed11_reward_v2_sf20_rrws_k24_training_ep10.pt"
FEATURES = ("precipitation_mm", "temperature_c", "relative_humidity_pct", "et0_mm",
            "vpd_kpa", "shortwave_radiation_w_m2")
FORECAST = ("forecast_precipitation_mm", "forecast_et0_mm", "forecast_temperature_c")
ASSET_LINE = re.compile(r"^const assets=(.*);$", re.MULTILINE)


def csv_rows(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def inputs():
    forecast = {row["timestamp"]: row for row in csv_rows("synthetic_forecast_sf20.csv")}
    weather = []
    start = datetime(2025, 1, 1)
    for row in csv_rows("benchmark_2025.csv"):
        stamp = row["timestamp"]
        if not "2025-01-01" <= stamp < "2025-01-08":
            continue
        now = start + timedelta(hours=len(weather))
        proxy = forecast[stamp]
        if (datetime.fromisoformat(stamp) != now or
                datetime.fromisoformat(proxy["forecast_target_timestamp"]) != now + timedelta(hours=1)):
            raise ValueError(f"Invalid hourly timestamp: {stamp}")
        weather.append([stamp, *(float(row[k]) for k in FEATURES), *(float(proxy[k]) for k in FORECAST)])
    if len(weather) != 168:
        raise ValueError("Expected 168 hourly inputs")
    norm = json.loads((DATA / "normalizer.json").read_text())
    return weather, [[norm[k]["mean"], norm[k]["std"]] for k in FEATURES]


def rebuild(actor_bytes):
    html = PAGE.read_text(encoding="utf-8")
    match = ASSET_LINE.search(html)
    names = json.loads(match[1])["weights"]
    weights = {name: base64.b64encode(actor_bytes(name)).decode("ascii") for name in names}
    weather, stats = inputs()
    assets = json.dumps(dict(weather=weather, stats=stats, weights=weights),
                        allow_nan=False, separators=(",", ":"))
    updated = html[:match.start(1)] + assets + html[match.end(1):]
    temporary = PAGE.with_suffix(".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(PAGE)


if __name__ == "__main__":
    import torch
    actor = torch.load(CHECKPOINT, weights_only=True, map_location="cpu")["actor"]
    rebuild(lambda name: actor[name].detach().numpy().astype("<f4").tobytes())
