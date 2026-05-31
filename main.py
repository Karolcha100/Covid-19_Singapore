import pandas as pd
import pathlib as pth
import argparse






from scripts.app import ForecastApp
from scripts.process_data_initial import process_data




def init_run() -> None:
    df: pd.DataFrame = process_data(pth.Path(f"data_raw/SG.csv"))
    
    df.to_csv(f"data_processed/SG_nona.csv")

    df = pd.read_csv(f"data_processed/SG_nona.csv", parse_dates=["date"])

    app = ForecastApp(
        df=df,
        default_date_min=0,
        default_date_max=730,
        default_date_pred=180,
        port=8050,
        debug=True,
    )
    app.run()



if __name__ == "__main__":
    init_run()