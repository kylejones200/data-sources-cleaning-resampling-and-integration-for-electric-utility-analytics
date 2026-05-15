from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot

np.random.seed(42)
signalplot.apply(font_family="serif")


@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"


def load_config(config_path=None) -> "Config":
    """Build Config from config.yaml, falling back to dataclass defaults."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        return Config()
    with open(config_path) as _f:
        import yaml as _yaml

        raw = _yaml.safe_load(_f) or {}
    _d = raw.get("data", {})
    _m = raw.get("model", {})
    _o = raw.get("output", {})
    return Config(
        csv_path=_d.get(
            "input_file",
            "2001-2025 Net_generation_United_States_all_sectors_monthly.csv",
        ),
        freq=_d.get("freq", "MS"),
    )


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s


def main(plot: bool = False):
    cfg = load_config()
    s = load_series(cfg)
    yoy = s.pct_change(12) * 100.0
    yearly = s.resample("Y").mean()

    if plot:
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
        ax[0].plot(s.index, s.values, label="EIA monthly")
        ax[0].legend()
        ax[1].bar(yearly.index.year, yearly.values, label="Yearly mean")
        ax[1].legend()
        ax[2].plot(yoy.index, yoy.values, color="tab:orange", label="YoY %")
        ax[2].axhline(0, color="k", lw=0.5)
        ax[2].legend()
        signalplot.save("eia_viz.png")


if __name__ == "__main__":
    main()
