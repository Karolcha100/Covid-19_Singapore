import pandas as pd
import pathlib as pth
import argparse
from scripts.app import ForecastApp
from scripts.process_data_initial import process_data
from scripts.visualisations_plotly import build_report

colors = {
    'CONFIRMED': '#52A929',
    'VACCINATED': '#00D5D2',
    'FULLY_VACCINATED': '#D500DA',
    'DECEASED': '#D50000',
    'CFR': '#D32F2F',
}

sg_events = {
    "Circuit Breaker": "2020-04-07",
    "Vaccination Starts": "2020-12-30",
    "Delta Wave": "2021-08-01",
    "Omicron Wave": "2021-12-15",
}

def init_run() -> None:
    df: pd.DataFrame = process_data(pth.Path(f"data_raw/SG.csv"))
    
    df.to_csv(f"data_processed/SG_nona.csv")

    df = pd.read_csv(f"data_processed/SG_nona.csv", parse_dates=["date"])

    build_report(df, color_dict=colors, event_dict=sg_events, figname="report.html")

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