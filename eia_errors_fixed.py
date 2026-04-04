import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit

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


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.copy()
    # Avoid div by zero
    yt[yt == 0] = np.finfo(float).eps
    return float(np.mean(np.abs((y_true - y_pred) / yt)) * 100.0)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = np.finfo(float).eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def mase_denom_seasonal(y: pd.Series, season: int) -> float:
    # Mean absolute seasonal difference y_t - y_{t-s}
    diffs = np.abs(y.values[season:] - y.values[:-season])
    if len(diffs) == 0 or np.all(diffs == 0):
        return 1.0
    return float(np.mean(diffs))


def rolling_origin_metrics(y: pd.Series, cfg: Config):
    idx = np.arange(len(y))
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    metrics = []
    last_true, last_pred = None, None
    for tr, te in tscv.split(idx):
        end = tr[-1]
        y_tr = y.iloc[: end + 1]
        y_te = y.iloc[end + 1 : end + 1 + cfg.horizon]
        if len(y_te) == 0:
            continue
        # Seasonal naive forecast
        yhat = []
        for i in range(len(y_te)):
            src_idx = end + 1 + i - cfg.season
            if src_idx >= 0:
                yhat.append(y.iloc[src_idx])
            else:
                yhat.append(y_tr.iloc[-1])
        yhat = np.asarray(yhat, dtype=float)
        # Metrics
        m_mae = mae(y_te.values, yhat)
        m_mape = mape(y_te.values, yhat)
        m_smape = smape(y_te.values, yhat)
        denom = mase_denom_seasonal(y_tr, cfg.season)
        m_mase = float(np.mean(np.abs(y_te.values - yhat)) / denom)
        metrics.append({
            'MAE': m_mae,
            'MAPE': m_mape,
            'SMAPE': m_smape,
            'MASE': m_mase,
        })
        last_true, last_pred = y_te, pd.Series(yhat, index=y_te.index)
    return metrics, last_true, last_pred


def main():
    cfg = Config()
    y = load_series(cfg)
    metrics, last_true, last_pred = rolling_origin_metrics(y, cfg)
    # Print mean metrics
    dfm = pd.DataFrame(metrics)
    print(dfm.mean().to_string())

    # Plot last fold
    plt.figure(figsize=(9,4))
    plt.plot(y.index, y.values, label='history', alpha=0.6)
    if last_pred is not None:
        plt.plot(last_pred.index, last_pred.values, label='Seasonal naive last fold')
    plt.legend()
    save_fig('eia_errors_last_fold.png')

if __name__ == '__main__':
    main()
