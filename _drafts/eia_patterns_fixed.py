from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot
from statsmodels.tsa.seasonal import seasonal_decompose

np.random.seed(42)
signalplot.apply(font_family="serif")


@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    season: int = 12


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
        season=_m.get("season", 12),
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
    dec = seasonal_decompose(s, model="additive", period=cfg.season)
    if plot:
        fig, ax = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
        ax[0].plot(s.index, s.values)
        ax[0].set_title("Observed")
        ax[1].plot(dec.trend.index, dec.trend.values)
        ax[1].set_title("Trend")
        ax[2].plot(dec.seasonal.index, dec.seasonal.values)
        ax[2].set_title("Seasonal")
        ax[3].plot(dec.resid.index, dec.resid.values)
        ax[3].set_title("Residual")
        signalplot.save("eia_patterns.png")
        # Seasonal subseries plot
        sub = s.copy()
        sub_df = sub.to_frame("value")
        sub_df["month"] = sub_df.index.month
        sub_df["year"] = sub_df.index.year
        plt.figure(figsize=(10, 6))
        for m in range(1, 13):
            part = sub_df[sub_df["month"] == m]
            plt.plot(part["year"], part["value"], label=f"M{m}", alpha=0.6)
        plt.legend(ncol=3, fontsize=8)
        plt.xlabel("Year")
        plt.ylabel("Value")
        signalplot.save("eia_seasonal_subseries.png")


if __name__ == "__main__":
    main()
