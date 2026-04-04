import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

np.random.seed(42)
plt.rcParams.update({'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "/Users/k.jones/Downloads/medium-export-e6bf40a8b01915d7380f6f547e0dd25ddd791328d4d9fa3a77513e82e662373c/posts/2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    season: int = 12
    drop_ratio: float = 0.05  # fraction of points to drop (simulate missing)


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s


def seasonal_mean_impute(s: pd.Series, season: int) -> pd.Series:
    # Compute seasonal means by month
    by_month = s.groupby(s.index.month).mean()
    filled = s.copy()
    na_idx = s.index[s.isna()]
    for ts in na_idx:
        filled.loc[ts] = by_month.loc[ts.month]
    return filled


def main():
    cfg = Config()
    s = load_series(cfg)
    # Simulate missing values
    n = len(s)
    k = max(1, int(cfg.drop_ratio * n))
    miss_idx = np.random.choice(np.arange(n), size=k, replace=False)
    s_miss = s.copy()
    s_miss.iloc[miss_idx] = np.nan

    # Impute
    s_seas = seasonal_mean_impute(s_miss, cfg.season)
    s_lin = s_miss.interpolate(method='time')

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(s.index, s.values, label='original', alpha=0.4)
    plt.scatter(s_miss.index[s_miss.isna()], s.loc[s_miss.isna()], color='red', s=18, label='removed')
    plt.plot(s_seas.index, s_seas.values, label='seasonal mean', alpha=0.8)
    plt.plot(s_lin.index, s_lin.values, label='time interpolation', alpha=0.8)
    plt.legend()
    save_fig('eia_impute_compare.png')

if __name__ == '__main__':
    main()
