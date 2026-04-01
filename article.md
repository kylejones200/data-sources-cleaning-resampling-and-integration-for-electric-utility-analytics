# Data sources, cleaning, resampling, and integration for Electric Utility Analytics Electric utilities are data-rich but often insight-poor. Every second,
power grids generate vast amounts of telemetry from sensors, meters...

::::::### Data sources, cleaning, resampling, and integration for Electric Utility Analytics 

Electric utilities are data-rich but often insight-poor. Every second,
power grids generate vast amounts of telemetry from sensors, meters, and
control systems. Smart meters report household consumption in 15-minute
intervals. SCADA systems collect voltages and currents across
substations and feeders. Phasor Measurement Units stream high-resolution
synchrophasor data. Enterprise Asset Management platforms house detailed
records of transformers, breakers, and other field equipment. Yet
despite these torrents of data, many utilities still rely on manual
processes, siloed systems, and static reports.

The root issue is fragmentation. Operational Technology (OT) systems
like SCADA are often isolated from Information Technology (IT)
environments that host enterprise and market data. AMI data may reside
in separate customer information systems. Maintenance records might be
buried in work order logs. Integrating these disparate streams is
cumbersome, often requiring bespoke ETL pipelines. As a result, much of
the data sits unused, limiting its value for analytics, machine
learning, and decision support.

This creates tangible business problems. Maintenance crews lack
predictive insights because equipment health data remains disconnected
from condition monitoring sensors. Grid operators cannot fully leverage
weather and demand data together to anticipate loading risks. Regulatory
compliance reporting is tedious because data for audits is scattered
across incompatible formats. The cost of inefficiency is high: missed
opportunities to optimize investments, reduce outages, and improve
customer satisfaction.

### The Analytics Solution: Preparing Data for Machine Learning
Analytics begins with data readiness. To make machine learning work for
utilities, data must be accessible, reliable, and modeled in ways that
reflect grid realities. This project focuses on the mechanics of
preparing utility data for analysis. We will address three fundamental
tasks.

First, data cleaning. Utility data is often noisy, containing gaps,
duplicates, or faulty readings. Sensors malfunction, meters fail, and
logs contain inconsistent timestamps. Cleaning requires handling missing
values, removing erroneous spikes, and reconciling mismatched units or
formats.

Second, resampling and alignment. Utility datasets operate at different
granularities: AMI data may be every 15 minutes, SCADA readings every 4
seconds, and weather data hourly. Aligning these time series to common
intervals allows joint analysis. This often involves aggregation
(summing sub-minute SCADA readings to hourly values) or interpolation
(filling short gaps in time series).

Third, feature integration. Meaningful analytics often emerges when
multiple datasets are combined. Weather impacts demand, asset age
influences failure rates, vegetation encroachment correlates with storm
outages. Joining these datasets requires careful handling of time zones,
coordinate systems for geospatial joins, and equipment identifiers
across systems.

By addressing these steps systematically, utilities can unlock the full
value of their data. Properly prepared datasets feed into machine
learning models that predict failures, forecast load, and support
data-driven investment planning.

### From Raw Records to Actionable Signals
A typical example is transformer monitoring. SCADA data may include
transformer load and oil temperature, while EAM holds the installation
date and maintenance history. By joining these, we can calculate
load-to-age stress factors, compare them across similar units, and flag
transformers at higher risk of failure. Without integrated data, such
insights remain invisible.

Another example is storm readiness. Outage records stored in OMS systems
can be combined with feeder vegetation data and historical weather
records. By cleaning and aligning these datasets, we can train models
that predict which circuits are most likely to fail during high winds.
This directly informs crew staging and vegetation management priorities.

These cases highlight a recurring theme: data silos hide patterns that
cross organizational boundaries. Preparing data for analytics is as much
about breaking down silos as it is about technical preprocessing.

### Transition to the Demo
In this demo, we will work with synthetic smart meter and SCADA datasets
to illustrate practical data preparation steps. You will:

- Load raw time series datasets and inspect their structure.
- Clean noisy data by detecting and correcting errors such as missing
  readings or sensor spikes.
- Resample data to common intervals suitable for modeling.
- Join multiple datasets into a unified view aligned by timestamp and
  identifier.

We will also visualize these transformations, showing how raw meter
readings and SCADA telemetry evolve into clean, analytics-ready time
series. This exercise mirrors the early stages of any utility data
project: wrangling heterogeneous, messy data into a usable form.

This demo lays the groundwork for everything that follows. Whether
forecasting load, predicting outages, or optimizing maintenance
schedules, the quality of insights depends on the quality of the
underlying data. By mastering data preparation in the utility context,
we establish the foundation on which machine learning models will be
built in later projects.

```python
"""
Chapter 2: Data in Power and Utilities
Loading, cleaning, and visualizing AMI (smart meter) and SCADA-like time series.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_smart_meter_data(file_path):
    """
    Load smart meter data.

    Args:
        file_path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Cleaned smart meter data.
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df.rename(columns={"consumption_kwh": "Consumption_kWh"})
    print(f"Smart meter data loaded: {df.shape[0]} rows")
    return df

def clean_and_resample(df):
    """
    Clean missing values and resample hourly.
    """
    df = df.set_index("timestamp").sort_index()
    df = df.resample("h").mean()
    df["Consumption_kWh"] = df["Consumption_kWh"].ffill()
    return df

def plot_consumption(df):
    """
    Plot hourly consumption.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["Consumption_kWh"], color="black")
    plt.xlabel("Time")
    plt.ylabel("Consumption (kWh)")
    plt.title("Hourly Smart Meter Consumption")
    plt.tight_layout()
    plt.savefig("chapter2_smart_meter_plot.png")
    plt.show()

def generate_synthetic_scada_data():
    """
    Generate synthetic SCADA-like grid frequency data.
    """
    time = pd.date_range("2022-01-01", periods=1440, freq="min")  # 1 day of minute data
    freq = 60 + np.random.normal(0, 0.02, size=1440)  # Nominal 60 Hz with noise
    return pd.DataFrame({"timestamp": time, "frequency_hz": freq})

def plot_scada(df):
    """
    Plot SCADA-like frequency data.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["frequency_hz"], color="black")
    plt.xlabel("Time")
    plt.ylabel("Frequency (Hz)")
    plt.title("Synthetic SCADA Grid Frequency")
    plt.tight_layout()
    plt.savefig("chapter2_scada_frequency.png")
    plt.show()

if __name__ == "__main__":
    # Example with synthetic smart meter data
    smart_meter_data = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=96, freq="15min"),
        "consumption_kwh": np.random.uniform(0.2, 1.5, size=96)
    })
    os.makedirs("data", exist_ok=True)
    smart_meter_data.to_csv("data/smart_meter_sample.csv", index=False)

    df_meter = load_smart_meter_data("data/smart_meter_sample.csv")
    df_meter = clean_and_resample(df_meter)
    plot_consumption(df_meter)

    # SCADA synthetic example
    df_scada = generate_synthetic_scada_data()
    plot_scada(df_scada)
```


::::::::::::::By [Kyle Jones](https://medium.com/@kyle-t-jones) on
[October 6, 2025](https://medium.com/p/4efb4d20ee01).

[Canonical
link](https://medium.com/@kyle-t-jones/data-sources-cleaning-resampling-and-integration-for-electric-utility-analytics-4efb4d20ee01)

Exported from [Medium](https://medium.com) on November 10, 2025.
