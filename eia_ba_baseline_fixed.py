import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

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


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def make_calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    m = df.index.month.values
    # sine/cos seasonal terms
    df['sin12'] = np.sin(2 * np.pi * m / 12.0)
    df['cos12'] = np.cos(2 * np.pi * m / 12.0)
    # month dummies
    for k in range(1, 13):
        df[f'm{k}'] = (m == k).astype(int)
    return df


def rolling_origin_linear(y: pd.Series, cfg: Config):
    idx = np.arange(len(y))
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    maes = []
    last_true, last_pred = None, None
    for tr, te in tscv.split(idx):
        end = tr[-1]
        y_tr = y.iloc[: end + 1]
        y_te = y.iloc[end + 1 : end + 1 + cfg.horizon]
        if len(y_te) == 0:
            continue
        X_tr = make_calendar_features(y_tr.index)
        X_te = make_calendar_features(y_te.index)
        model = LinearRegression()
        model.fit(X_tr, y_tr.values)
        yhat = model.predict(X_te)
        maes.append(mean_absolute_error(y_te.values, yhat))
        last_true = y_te
        last_pred = pd.Series(yhat, index=y_te.index)
    return float(np.mean(maes)), last_true, last_pred


def main():
    cfg = Config()
    y = load_series(cfg)
    mean_mae, y_true, y_pred = rolling_origin_linear(y, cfg)
    print(f"Linear calendar baseline mean MAE: {mean_mae}")

    plt.figure(figsize=(9,4))
    plt.plot(y.index, y.values, label='history', alpha=0.6)
    if y_pred is not None:
        plt.plot(y_pred.index, y_pred.values, label='Linear baseline last fold')
    plt.legend()
    save_fig('eia_ba_baseline.png')

if __name__ == '__main__':
    main()
