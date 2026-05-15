"""Generated from Jupyter notebook: Ercot combined data for overall demand to 2022

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import pandas as pd


def main():
    # --- code cell ---

    df = pd.read_csv("/Users/jnesnky/ercot/combinedfile.csv")


    # --- code cell ---

    df.head()


    # --- code cell ---

    df["end"] = pd.to_datetime(df["end"])


    # --- code cell ---

    df.set_index("end", inplace=True)


    # --- code cell ---

    df.to_csv("/Users/jnesnky/ercot/combinedfile_datetime.csv")


    # --- code cell ---

    df["ERCOT"].plot()


    # --- code cell ---

    # %matplotlib inline  # Jupyter-only


if __name__ == "__main__":
    main()
