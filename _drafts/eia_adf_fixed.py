import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
np.random.seed(42)
signalplot.apply(font_family="serif")


@dataclass
class Config:
    csv_path: str = "/Users/k.jones/Downloads/medium-export-e6bf40a8b01915d7380f6f547e0dd25ddd791328d4d9fa3a77513e82e662373c/posts/2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
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
            "/Users/k.jones/Downloads/medium-export-e6bf40a8b01915d7380f6f547e0dd25ddd791328d4d9fa3a77513e82e662373c/posts/2001-2025 Net_generation_United_States_all_sectors_monthly.csv",
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


def adf_summary(y: pd.Series) -> dict:
    res = adfuller(y.dropna().values, autolag="AIC")
    keys = ["ADF Statistic", "p-value", "lags used", "nobs"]
    out = dict(zip(keys, res[:4]))
    return out


def main(plot: bool = False):
    cfg = load_config()
    s = load_series(cfg)
    base = adf_summary(s)

    sd = s.diff(cfg.season).dropna()
    seas = adf_summary(sd)

    logger.info("ADF on raw:", base)
    logger.info("ADF on seasonal-differenced:", seas)

    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        ax[0, 0].plot(s.index, s.values)
        ax[0, 0].set_title("EIA series")
        ax[0, 1].plot(sd.index, sd.values)
        ax[0, 1].set_title("Seasonal diff (12)")
        plot_acf(sd.dropna(), ax=ax[1, 0], lags=24)
        plot_pacf(sd.dropna(), ax=ax[1, 1], lags=24, method="ywm")
        signalplot.save("eia_adf.png")


if __name__ == "__main__":
    main()
