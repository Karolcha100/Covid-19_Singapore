import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _add_event_lines(fig: go.Figure, event_dict: dict | None,
                     rows: list[int], cols: list[int],
                     is_subplot: bool = True) -> None:
    """
    Add vertical dashed lines + annotations for each event.

    Parameters
    ----------
    fig        : plotly Figure
    event_dict : {label: date_string}
    rows, cols : which subplot cells to annotate (paired lists, 1-indexed)
    is_subplot : True if figure was created with make_subplots, False for go.Figure()
    """
    if not event_dict:
        return

    for event_name, date_str in event_dict.items():
        date_obj = pd.to_datetime(date_str)

        for row, col in zip(rows, cols):
            # add_vline accepts row/col only on subplot figures
            vline_kwargs = dict(
                x=date_obj.timestamp() * 1000,
                line=dict(color="dimgray", dash="dash", width=1.5),
                opacity=0.7,
            )
            if is_subplot:
                vline_kwargs.update(row=row, col=col)
            fig.add_vline(**vline_kwargs)

            # add_annotation with row/col requires a subplot grid —
            # for plain go.Figure() we reference axes directly via yref/xref.
            if is_subplot:
                fig.add_annotation(
                    x=date_obj,
                    y=1.0,
                    yref=f"y{_axis_id(row, col)} domain",
                    text=f" {event_name}",
                    showarrow=False,
                    textangle=-90,
                    xanchor="right",
                    yanchor="top",
                    font=dict(size=9, color="dimgray"),
                    row=row, col=col,
                )
            else:
                fig.add_annotation(
                    x=date_obj,
                    y=1.0,
                    xref="x",
                    yref="y domain",
                    text=f" {event_name}",
                    showarrow=False,
                    textangle=-90,
                    xanchor="right",
                    yanchor="top",
                    font=dict(size=9, color="dimgray"),
                )


def _axis_id(row: int, col: int, ncols: int = 2) -> str:
    """Convert (row, col) 1-indexed to plotly yaxis suffix string."""
    idx = (row - 1) * ncols + col
    return "" if idx == 1 else str(idx)


def _xaxis_opts() -> dict:
    """Common x-axis formatting options."""
    return dict(
        tickformat="%Y-%m-%d",
        dtick="M3",
        tickangle=45,
        tickfont=dict(size=7),
        showgrid=True,
        gridcolor="#D5D5D5",
    )

def simple_descriptive_plots_grid(
    df: pd.DataFrame,
    color_dict: dict,
    event_dict: dict | None = None,
) -> go.Figure:
    """
    2×2 dashboard.
      [0,0] Daily Cases & Deaths          [0,1] Cumulative Cases & Deaths
      [1,0] Daily Vaccinations            [1,1] Cumulative Vaccinations (%)

    Returns a plotly Figure (call .show() or .write_html()).
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Daily Epidemic Dynamics",
            "Cumulative Epidemic Totals",
            "Daily Vaccination Progress",
            "Cumulative Vaccination (% of Population)",
        ],
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    roll7 = lambda s: s.rolling(7).mean()

    # ── [1,1] Daily Cases & Deaths ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["new_confirmed"],
        mode="markers", marker=dict(color=color_dict["CONFIRMED"], opacity=0.15, size=4),
        name="Daily Confirmed", legendgroup="confirmed", showlegend=True,
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=roll7(df["new_confirmed"]),
        mode="lines", line=dict(color=color_dict["CONFIRMED"], width=2),
        name="Confirmed (7d avg)", legendgroup="confirmed7",
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["new_deceased"],
        mode="markers", marker=dict(color=color_dict["DECEASED"], opacity=0.15, size=4),
        name="Daily Deaths", legendgroup="deceased",
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df["date"], y=roll7(df["new_deceased"]),
        mode="lines", line=dict(color=color_dict["DECEASED"], width=2),
        name="Deaths (7d avg)", legendgroup="deceased7",
    ), row=1, col=1, secondary_y=True)

    fig.update_yaxes(title_text="Daily Cases",  color=color_dict["CONFIRMED"], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Daily Deaths", color=color_dict["DECEASED"],  row=1, col=1, secondary_y=True)

    # ── [1,2] Cumulative Cases & Deaths ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_confirmed"],
        mode="lines", line=dict(color=color_dict["CONFIRMED"], width=2),
        name="Cum. Cases",
    ), row=1, col=2, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_deceased"],
        mode="lines", line=dict(color=color_dict["DECEASED"], width=2),
        name="Cum. Deaths",
    ), row=1, col=2, secondary_y=True)

    fig.update_yaxes(title_text="Total Cases",  color=color_dict["CONFIRMED"], row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Total Deaths", color=color_dict["DECEASED"],  row=1, col=2, secondary_y=True)

    # ── [2,1] Daily Vaccinations ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["new_persons_vaccinated"],
        mode="markers", marker=dict(color=color_dict["VACCINATED"], opacity=0.15, size=4),
        name="First Dose",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["date"], y=roll7(df["new_persons_vaccinated"]),
        mode="lines", line=dict(color=color_dict["VACCINATED"], width=2),
        name="1st Dose (7d avg)",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["new_persons_fully_vaccinated"],
        mode="markers", marker=dict(color=color_dict["FULLY_VACCINATED"], opacity=0.15, size=4),
        name="Fully Vacc.",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["date"], y=roll7(df["new_persons_fully_vaccinated"]),
        mode="lines", line=dict(color=color_dict["FULLY_VACCINATED"], width=2),
        name="Fully Vacc. (7d avg)",
    ), row=2, col=1)

    fig.update_yaxes(title_text="Doses Given Per Day", row=2, col=1)

    # ── [2,2] Cumulative Vaccinations % ─────────────────────────────────────
    perc_vacc = (df["cumulative_persons_vaccinated"] / df["population"]) * 100
    perc_full = (df["cumulative_persons_fully_vaccinated"] / df["population"]) * 100

    fig.add_trace(go.Scatter(
        x=df["date"], y=perc_vacc,
        mode="lines", line=dict(color=color_dict["VACCINATED"], width=2),
        name="Cum. 1st Dose (%)",
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=df["date"], y=perc_full,
        mode="lines", line=dict(color=color_dict["FULLY_VACCINATED"], width=2),
        name="Cum. Fully Vacc. (%)",
    ), row=2, col=2)

    fig.update_yaxes(title_text="Percentage of Population (%)", row=2, col=2)

    # ── Event lines on all subplots ──────────────────────────────────────────
    _add_event_lines(fig, event_dict,
                     rows=[1, 1, 2, 2],
                     cols=[1, 2, 1, 2])

    # ── Global layout ────────────────────────────────────────────────────────
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(_xaxis_opts(), row=r, col=c)

    fig.update_layout(
        height=900, width=1200,
        title_text="COVID-19 Dashboard",
        legend=dict(orientation="v", x=1.02, y=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def cumulative_totals_plot(
    df: pd.DataFrame,
    color_dict: dict,
    event_dict: dict | None = None,
) -> go.Figure:
    """
    Standalone plot: cumulative cases + vaccinations (left Y) vs deaths (right Y).
    Returns a plotly Figure.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_confirmed"],
        mode="lines", line=dict(color=color_dict["CONFIRMED"], width=2),
        name="Cum. Confirmed",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_persons_vaccinated"],
        mode="lines", line=dict(color=color_dict["VACCINATED"], width=2),
        name="Cum. Vaccinated",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_persons_fully_vaccinated"],
        mode="lines", line=dict(color=color_dict["FULLY_VACCINATED"], width=2),
        name="Cum. Fully Vacc.",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_deceased"],
        mode="lines", line=dict(color=color_dict["DECEASED"], width=2),
        name="Cum. Deaths",
    ), secondary_y=True)

    _add_event_lines(fig, event_dict, rows=[1], cols=[1])

    fig.update_xaxes(_xaxis_opts())
    fig.update_yaxes(title_text="Total Cases / Vaccinations", secondary_y=False)
    fig.update_yaxes(title_text="Total Deaths", color=color_dict["DECEASED"], secondary_y=True)

    fig.update_layout(
        height=550, width=1000,
        title_text="Overall Cumulative Totals: Cases, Vaccinations, and Deaths",
        legend=dict(x=1.08, y=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def cfr_plot(
    df: pd.DataFrame,
    color_dict: dict,
    event_dict: dict | None = None,
) -> go.Figure:
    """
    14-day lagged Case Fatality Rate using 7-day rolling averages.
    CFR_t = (Deaths_t / Cases_{t-14}) × 100
    Returns a plotly Figure.
    """
    smooth_cases  = df["new_confirmed"].rolling(7).mean()
    smooth_deaths = df["new_deceased"].rolling(7).mean()
    lagged_cases  = smooth_cases.shift(14)

    cfr_series = (smooth_deaths / lagged_cases) * 100
    cfr_series = cfr_series.replace([np.inf, -np.inf], np.nan)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=cfr_series,
        mode="lines",
        line=dict(color=color_dict["CFR"], width=2.5),
        name="Lagged CFR (7d avg)",
    ))

    _add_event_lines(fig, event_dict, rows=[1], cols=[1], is_subplot=False)

    fig.update_xaxes(_xaxis_opts())
    fig.update_yaxes(title_text="Case Fatality Rate (%)", showgrid=True, gridcolor="#D5D5D5")

    fig.update_layout(
        height=500, width=1000,
        title_text="Lagged Case Fatality Rate (CFR) Dynamics",
        legend=dict(x=0.98, y=0.98, xanchor="right"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def build_report(
    df: pd.DataFrame,
    color_dict: dict,
    event_dict: dict | None = None,
    figname: str = "report.html",
) -> None:
    """
    Generates all three interactive figures and saves them as a single
    self-contained HTML file.

    Parameters
    ----------
    df         : processed DataFrame with date column already parsed
    color_dict : colour mapping dict (same keys as in original script)
    event_dict : {label: date_string} dict or None
    figname    : output path for the HTML report
    """
    fig_grid  = simple_descriptive_plots_grid(df, color_dict, event_dict)
    fig_cumul = cumulative_totals_plot(df, color_dict, event_dict)
    fig_cfr   = cfr_plot(df, color_dict, event_dict)

    # Render each figure to an HTML div (no full-page boilerplate, no CDN script
    # for the first two — we'll inject the CDN script once manually)
    grid_div  = fig_grid.to_html(full_html=False, include_plotlyjs=False)
    cumul_div = fig_cumul.to_html(full_html=False, include_plotlyjs=False)
    cfr_div   = fig_cfr.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>COVID-19 Interactive Report</title>
  <!-- Plotly CDN — single load, shared by all figures -->
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg:      #f5f5f0;
      --card:    #ffffff;
      --border:  #e0ddd5;
      --accent:  #1a1a2e;
      --muted:   #888880;
      --radius:  6px;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      background: var(--bg);
      font-family: 'Georgia', serif;
      color: var(--accent);
      padding: 2rem 1.5rem 4rem;
    }}

    header {{
      max-width: 1280px;
      margin: 0 auto 2.5rem;
      border-bottom: 2px solid var(--accent);
      padding-bottom: 1rem;
    }}

    header h1 {{
      font-size: 2rem;
      letter-spacing: 0.02em;
      font-weight: normal;
    }}

    header p {{
      margin-top: 0.4rem;
      color: var(--muted);
      font-size: 0.95rem;
      font-style: italic;
    }}

    .section {{
      max-width: 1280px;
      margin: 0 auto 2.5rem;
    }}

    .section h2 {{
      font-size: 1.1rem;
      font-weight: normal;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin-bottom: 0.8rem;
      padding-left: 0.15rem;
    }}

    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem 1rem 0.5rem;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}

    footer {{
      max-width: 1280px;
      margin: 0 auto;
      font-size: 0.8rem;
      color: var(--muted);
      text-align: center;
      font-style: italic;
    }}
  </style>
</head>
<body>

<header>
  <h1>COVID-19 Epidemiological Report</h1>
  <p>Interactive visualisations — hover to inspect values, drag to zoom, double-click to reset.</p>
</header>

<div class="section">
  <h2>Dashboard — Cases, Deaths &amp; Vaccinations</h2>
  <div class="card">{grid_div}</div>
</div>

<div class="section">
  <h2>Cumulative Totals Overview</h2>
  <div class="card">{cumul_div}</div>
</div>

<div class="section">
  <h2>Case Fatality Rate (14-day lag)</h2>
  <div class="card">{cfr_div}</div>
</div>

<footer>Generated with Plotly · All figures share a linked Plotly.js bundle.</footer>

</body>
</html>"""

    with open(figname, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved → {figname}")


data = pd.read_csv('../data_processed/SG_nona.csv')
data['date'] = pd.to_datetime(data['date'])

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

build_report(data, color_dict=colors, event_dict=sg_events, figname="report.html")
