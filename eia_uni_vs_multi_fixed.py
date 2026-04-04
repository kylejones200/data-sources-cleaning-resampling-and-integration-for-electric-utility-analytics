import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

np.random.seed(42)
plt.rcParams.update({'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5
    season: int = 12


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def make_exog(idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    m = df.index.month.values
    df['sin12'] = np.sin(2 * np.pi * m / 12.0)
    df['cos12'] = np.cos(2 * np.pi * m / 12.0)
    # month dummies
    for k in range(1, 13):
        df[f'm{k}'] = (m == k).astype(int)
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
        uni = sm.tsa.statespace.SARIMAX(y_tr, order=(1,1,1), seasonal_order=(1,1,1,cfg.season), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        f_uni = uni.forecast(len(y_te)).to_numpy()
        uni_maes.append(mean_absolute_error(y_te.values, f_uni))
        # Multivariate with exogenous calendar features
        X_tr = make_exog(y_tr.index)
        X_te = make_exog(y_te.index)
        mul = sm.tsa.statespace.SARIMAX(y_tr, exog=X_tr, order=(1,1,1), seasonal_order=(1,1,1,cfg.season), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        f_mul = mul.forecast(len(y_te), exog=X_te).to_numpy()
        mul_maes.append(mean_absolute_error(y_te.values, f_mul))
        last = {
            'true': y_te,
            'Univariate': pd.Series(f_uni, index=y_te.index),
            'WithExog': pd.Series(f_mul, index=y_te.index),
        }
    return float(np.mean(uni_maes)), float(np.mean(mul_maes)), last


def main():
    cfg = Config()
    y = load_series(cfg)
    uni_m, mul_m, last = rolling_origin_compare(y, cfg)
    print(f"SARIMAX univariate mean MAE: {uni_m}")
    print(f"SARIMAX with exogenous calendar features mean MAE: {mul_m}")

    plt.figure(figsize=(9,4))
    plt.plot(y.index, y.values, label='history', alpha=0.6)
    if last:
        for name in ['Univariate','WithExog']:
            plt.plot(last[name].index, last[name].values, label=f"{name} last fold")
    plt.legend()
    save_fig('eia_uni_vs_multi_last_fold.png')

if __name__ == '__main__':
    main()
