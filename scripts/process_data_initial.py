import pandas as pd
import pathlib as pth


def process_data(path: pth.Path) -> pd.DataFrame:
    df_raw = pd.read_csv(path)

    df_raw["date"] = pd.to_datetime(df_raw["date"])

    return (
        df_raw
        .select_dtypes(exclude='string')
        .set_index("date")
        .drop(columns=["openstreetmap_id", "latitude", "longitude", "aggregation_level"])
        .interpolate(method="linear")
        .fillna(0)
    )