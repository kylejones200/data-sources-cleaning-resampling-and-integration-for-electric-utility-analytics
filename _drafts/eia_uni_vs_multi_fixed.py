import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
np.random.seed(42)
signalplot.apply(font_family="serif")


@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5
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
        horizon=_m.get("horizon", 12),
        n_splits=_d.get("n_splits", 5),
        season=_m.get("season", 12),
    )


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def make_exog(idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    m = df.index.month.values
    df["sin12"] = np.sin(2 * np.pi * m / 12.0)
    df["cos12"] = np.cos(2 * np.pi * m / 12.0)
    # month dummies
    for k in range(1, 13):
        df[f"m{k}"] = (m == k).astype(int)
    return df


def rolling_origin_compare(y: pd.Series, cfg: Config):
    idx = np.arange(len(y))
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    uni_maes, mul_maes = [], []
    last = {}
    for tr, te in tscv.split(idx):
        end = tr[-1]
        y_tr = y.iloc[: end + 1]
        y_te = y.iloc[end + 1 : end + 1 + cfg.horizon]
        if len(y_te) == 0:
            continue
        # Univariate SARIMAX
        uni = sm.tsa.statespace.SARIMAX(
            y_tr,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, cfg.season),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        f_uni = uni.forecast(len(y_te)).to_numpy()
        uni_maes.append(mean_absolute_error(y_te.values, f_uni))
        # Multivariate with exogenous calendar features
        X_tr = make_exog(y_tr.index)
        X_te = make_exog(y_te.index)
        mul = sm.tsa.statespace.SARIMAX(
            y_tr,
            exog=X_tr,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, cfg.season),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        f_mul = mul.forecast(len(y_te), exog=X_te).to_numpy()
        mul_maes.append(mean_absolute_error(y_te.values, f_mul))
        last = {
            "true": y_te,
            "Univariate": pd.Series(f_uni, index=y_te.index),
            "WithExog": pd.Series(f_mul, index=y_te.index),
        }
    return float(np.mean(uni_maes)), float(np.mean(mul_maes)), last


def main(plot: bool = False):
    cfg = load_config()
    y = load_series(cfg)
    uni_m, mul_m, last = rolling_origin_compare(y, cfg)
    logger.info(f"SARIMAX univariate mean MAE: {uni_m}")
    logger.info(f"SARIMAX with exogenous calendar features mean MAE: {mul_m}")

    if plot:
        plt.figure(figsize=(9, 4))
        plt.plot(y.index, y.values, label="history", alpha=0.6)
        if last:
            for name in ["Univariate", "WithExog"]:
                plt.plot(last[name].index, last[name].values, label=f"{name} last fold")
        plt.legend()
        signalplot.save("eia_uni_vs_multi_last_fold.png")


if __name__ == "__main__":
    main()
