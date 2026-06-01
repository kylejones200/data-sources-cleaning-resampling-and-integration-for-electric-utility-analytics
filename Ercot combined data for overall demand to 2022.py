"""Generated from Jupyter notebook: Ercot combined data for overall demand to 2022

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import pandas as pd


def main():
    df = pd.read_csv("data/combinedfile.csv")
    df.head()
    df["end"] = pd.to_datetime(df["end"])
    df.set_index("end", inplace=True)
    df.to_csv("data/combinedfile_datetime.csv")
    df["ERCOT"].plot()


def main() -> None:
    main()


if __name__ == "__main__":
    main()
