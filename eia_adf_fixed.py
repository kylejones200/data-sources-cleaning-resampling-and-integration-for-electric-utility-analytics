import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(42)
plt.rcParams.update({'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "/Users/k.jones/Downloads/medium-export-e6bf40a8b01915d7380f6f547e0dd25ddd791328d4d9fa3a77513e82e662373c/posts/2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    season: int = 12


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s


def adf_summary(y: pd.Series) -> dict:
    res = adfuller(y.dropna().values, autolag='AIC')
    keys = ['ADF Statistic','p-value','lags used','nobs']
    out = dict(zip(keys, res[:4]))
    return out


def main():
    cfg = Config()
    s = load_series(cfg)
    base = adf_summary(s)

    sd = s.diff(cfg.season).dropna()
    seas = adf_summary(sd)

    print("ADF on raw:", base)
    print("ADF on seasonal-differenced:", seas)

    fig, ax = plt.subplots(2, 2, figsize=(10,6))
    ax[0,0].plot(s.index, s.values); ax[0,0].set_title('EIA series')
    ax[0,1].plot(sd.index, sd.values); ax[0,1].set_title('Seasonal diff (12)')
    plot_acf(sd.dropna(), ax=ax[1,0], lags=24)
    plot_pacf(sd.dropna(), ax=ax[1,1], lags=24, method='ywm')
    save_fig('eia_adf.png')

if __name__ == "__main__":
    main()
