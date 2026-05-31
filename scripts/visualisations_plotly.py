import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _add_event_lines(
    fig: go.Figure,
    event_dict: dict | None,
    rows: list[int],
    cols: list[int],
    is_subplot: bool = True,
) -> None:
    """
    Vertical dashed lines + rotated labels for significant events.

    Parameters
    ----------
    fig        : plotly Figure
    event_dict : {label: date_string}
    rows, cols : subplot cells to annotate (paired lists, 1-indexed)
    is_subplot : False when fig is a plain go.Figure() (no make_subplots grid)
    """
    if not event_dict:
        return

    for event_name, date_str in event_dict.items():
        date_obj = pd.to_datetime(date_str)

        for row, col in zip(rows, cols):
            vline_kwargs = dict(
                x=date_obj.timestamp() * 1000,
                line=dict(color="dimgray", dash="dash", width=1.5),
                opacity=0.7,
            )
            if is_subplot:
                vline_kwargs.update(row=row, col=col)
            fig.add_vline(**vline_kwargs)

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
    """(row, col) 1-indexed → plotly yaxis suffix string."""
    idx = (row - 1) * ncols + col
    return "" if idx == 1 else str(idx)


def _xaxis_opts() -> dict:
    """Shared x-axis formatting: quarterly ticks, rotated labels, grid."""
    return dict(
        tickformat="%Y-%m-%d",
        dtick="M3",
        tickangle=45,
        tickfont=dict(size=7),
        showgrid=True,
        gridcolor="#D5D5D5",
    )


def _yaxis_opts() -> dict:
    """Shared y-axis formatting: horizontal grid."""
    return dict(
        showgrid=True,
        gridcolor="#D5D5D5",
        title_standoff=8,   # ← brings axis title closer to tick labels
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public figure-building functions
# ═══════════════════════════════════════════════════════════════════════════════

def simple_descriptive_plots_grid(
    df: pd.DataFrame,
    color_dict: dict,
    event_dict: dict | None = None,
) -> go.Figure:
    """
    2×2 interactive dashboard — each subplot has its own legend.

      [1,1] Daily Cases & Deaths       [1,2] Cumulative Cases & Deaths
      [2,1] Daily Vaccinations         [2,2] Cumulative Vaccinations (%)
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
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    roll7 = lambda s: s.rolling(7).mean()

    # ── [1,1]  Daily Cases & Deaths ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["new_confirmed"],
        mode="markers",
        marker=dict(color=color_dict["CONFIRMED"], opacity=0.15, size=4),
        name="Daily Confirmed",
        legendgroup="g11", legendgrouptitle_text="Daily Epidemic",
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=roll7(df["new_confirmed"]),
        mode="lines", line=dict(color=color_dict["CONFIRMED"], width=2),
        name="Confirmed (7d avg)",
        legendgroup="g11",
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["new_deceased"],
        mode="markers",
        marker=dict(color=color_dict["DECEASED"], opacity=0.15, size=4),
        name="Daily Deaths",
        legendgroup="g11",
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df["date"], y=roll7(df["new_deceased"]),
        mode="lines", line=dict(color=color_dict["DECEASED"], width=2),
        name="Deaths (7d avg)",
        legendgroup="g11",
    ), row=1, col=1, secondary_y=True)

    fig.update_yaxes(title_text="Daily Cases",  color=color_dict["CONFIRMED"],
                     **_yaxis_opts(), row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Daily Deaths", color=color_dict["DECEASED"],
                     showgrid=False, title_standoff=8,
                     row=1, col=1, secondary_y=True)

    # ── [1,2]  Cumulative Cases & Deaths ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_confirmed"],
        mode="lines", line=dict(color=color_dict["CONFIRMED"], width=2),
        name="Cum. Cases",
        legendgroup="g12", legendgrouptitle_text="Cumulative Epidemic",
    ), row=1, col=2, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_deceased"],
        mode="lines", line=dict(color=color_dict["DECEASED"], width=2),
        name="Cum. Deaths",
        legendgroup="g12",
    ), row=1, col=2, secondary_y=True)

    fig.update_yaxes(title_text="Total Cases",  color=color_dict["CONFIRMED"],
                     **_yaxis_opts(), row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Total Deaths", color=color_dict["DECEASED"],
                     showgrid=False, title_standoff=8,
                     row=1, col=2, secondary_y=True)

    # ── [2,1]  Daily Vaccinations ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["new_persons_vaccinated"],
        mode="markers",
        marker=dict(color=color_dict["VACCINATED"], opacity=0.15, size=4),
        name="First Dose",
        legendgroup="g21", legendgrouptitle_text="Daily Vaccination",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["date"], y=roll7(df["new_persons_vaccinated"]),
        mode="lines", line=dict(color=color_dict["VACCINATED"], width=2),
        name="1st Dose (7d avg)",
        legendgroup="g21",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["new_persons_fully_vaccinated"],
        mode="markers",
        marker=dict(color=color_dict["FULLY_VACCINATED"], opacity=0.15, size=4),
        name="Fully Vacc.",
        legendgroup="g21",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["date"], y=roll7(df["new_persons_fully_vaccinated"]),
        mode="lines", line=dict(color=color_dict["FULLY_VACCINATED"], width=2),
        name="Fully Vacc. (7d avg)",
        legendgroup="g21",
    ), row=2, col=1)

    fig.update_yaxes(title_text="Doses Given Per Day",
                     **_yaxis_opts(), row=2, col=1)

    # ── [2,2]  Cumulative Vaccinations % ─────────────────────────────────────
    perc_vacc = (df["cumulative_persons_vaccinated"] / df["population"]) * 100
    perc_full = (df["cumulative_persons_fully_vaccinated"] / df["population"]) * 100

    fig.add_trace(go.Scatter(
        x=df["date"], y=perc_vacc,
        mode="lines", line=dict(color=color_dict["VACCINATED"], width=2),
        name="Cum. 1st Dose (%)",
        legendgroup="g22", legendgrouptitle_text="Cumulative Vaccination",
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=df["date"], y=perc_full,
        mode="lines", line=dict(color=color_dict["FULLY_VACCINATED"], width=2),
        name="Cum. Fully Vacc. (%)",
        legendgroup="g22",
    ), row=2, col=2)

    fig.update_yaxes(title_text="Percentage of Population (%)",
                     **_yaxis_opts(), row=2, col=2)

    # ── Event lines ──────────────────────────────────────────────────────────
    _add_event_lines(fig, event_dict,
                     rows=[1, 1, 2, 2],
                     cols=[1, 2, 1, 2])

    # ── X-axes + layout ───────────────────────────────────────────────────────
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(_xaxis_opts(), row=r, col=c)

    # Four separate legend boxes — centre-left of each subplot quadrant
    fig.update_layout(
        height=920, width=1220,
        title_text="COVID-19 Dashboard",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            x=0.02, y=0.75,          # centre-left of subplot [1,1]
            xanchor="left", yanchor="middle",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
            groupclick="toggleitem",
        ),
        legend2=dict(
            x=0.55, y=0.75,          # centre-left of subplot [1,2]
            xanchor="left", yanchor="middle",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
            groupclick="toggleitem",
        ),
        legend3=dict(
            x=0.02, y=0.22,          # centre-left of subplot [2,1]
            xanchor="left", yanchor="middle",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
            groupclick="toggleitem",
        ),
        legend4=dict(
            x=0.55, y=0.22,          # centre-left of subplot [2,2]
            xanchor="left", yanchor="middle",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
            groupclick="toggleitem",
        ),
    )

    # Assign each legendgroup to its dedicated legend box
    for trace in fig.data:
        lg = trace.legendgroup
        if lg == "g11":
            trace.legend = "legend"
        elif lg == "g12":
            trace.legend = "legend2"
        elif lg == "g21":
            trace.legend = "legend3"
        elif lg == "g22":
            trace.legend = "legend4"

    return fig


def cumulative_totals_plot(
    df: pd.DataFrame,
    color_dict: dict,
    event_dict: dict | None = None,
) -> go.Figure:
    """
    Standalone: cumulative cases + vaccinations (left Y) vs deaths (right Y).
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
    fig.update_yaxes(title_text="Total Cases / Vaccinations",
                     **_yaxis_opts(), secondary_y=False)
    fig.update_yaxes(title_text="Total Deaths", color=color_dict["DECEASED"],
                     showgrid=False, title_standoff=8, secondary_y=True)

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
    fig.update_yaxes(title_text="Case Fatality Rate (%)",
                     **_yaxis_opts())

    fig.update_layout(
        height=500, width=1000,
        title_text="Lagged Case Fatality Rate (CFR) Dynamics",
        legend=dict(x=0.98, y=0.98, xanchor="right"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# HTML report builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_stat_cards_html(stats: dict | None) -> str:
    """
    Render a row of KPI stat-cards from a dict:
        { "Label": ("value", "sub-text", "#hexcolor"), ... }
    Returns an empty string when stats is None.
    """
    if not stats:
        return ""

    cards = "\n".join(
        f"""  <div class="stat-card">
    <div class="stat-value" style="color:{color};">{value}</div>
    <div class="stat-label">{label}</div>
    <div class="stat-sub">{sub}</div>
  </div>"""
        for label, (value, sub, color) in stats.items()
    )

    return f'<div class="stat-row">\n{cards}\n</div>'



def _img_to_base64(path: str) -> str:
    """Читает файл изображения и возвращает data-URI для вставки в <img src=>."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "svg": "image/svg+xml"}.get(ext, "image/png")
    return f"data:{mime};base64,{data}"

def build_report(
    df: pd.DataFrame,
    color_dict: dict,
    event_dict: dict | None = None,
    stats: dict | None = None,
    images: list[dict] | None = None,
    figname: str = "report.html",
) -> None:
    """
    Combine all figures into a single self-contained HTML report.

    Parameters
    ----------
    df         : DataFrame with a parsed ``date`` column
    color_dict : colour mapping (same keys as the plotting functions)
    event_dict : {label: date_string} or None
    stats      : {label: (value, sub_text, hex_color)} KPI cards, or None
    images     : list of {"path": ..., "caption": ...} dicts, or None
    figname    : output file path
    """

    # ── 1. Build figures ─────────────────────────────────────────────────────
    fig_grid  = simple_descriptive_plots_grid(df, color_dict, event_dict)
    fig_cumul = cumulative_totals_plot(df, color_dict, event_dict)
    fig_cfr   = cfr_plot(df, color_dict, event_dict)

    grid_div  = fig_grid.to_html(full_html=False, include_plotlyjs=False)
    cumul_div = fig_cumul.to_html(full_html=False, include_plotlyjs=False)
    cfr_div   = fig_cfr.to_html(full_html=False, include_plotlyjs=False)

    cases_src = _img_to_base64("../plots/confirmed_pred.png")
    deaths_src = _img_to_base64("../plots/deceased_pred.png")

    # ── 2. Optional sections ─────────────────────────────────────────────────
    stat_cards_html = _build_stat_cards_html(stats)

    # ── 3. Assemble HTML in named parts ──────────────────────────────────────

    html_head = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>COVID-19 Interactive Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
"""

    html_styles = """\
  <style>
    :root {
      --bg:      #f5f5f0;
      --card:    #ffffff;
      --border:  #e0ddd5;
      --accent:  #1a1a2e;
      --muted:   #888880;
      --radius:  6px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      font-family: 'Georgia', serif;
      color: var(--accent);
      padding: 2rem 1.5rem 4rem;
    }

    /* ── Header ── */
    header {
      max-width: 1280px;
      margin: 0 auto 2rem;
      border-bottom: 2px solid var(--accent);
      padding-bottom: 1rem;
    }
    header h1 { font-size: 2rem; letter-spacing: 0.02em; font-weight: normal; }
    header p  { margin-top: 0.4rem; color: var(--muted); font-size: 0.95rem; font-style: italic; }

    /* ── Sections & cards ── */
    .section { max-width: 1280px; margin: 0 auto 2.5rem; }
    .section h2 {
      font-size: 1rem; font-weight: normal;
      text-transform: uppercase; letter-spacing: 0.12em;
      color: var(--muted); margin-bottom: 0.8rem;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem 1rem 0.5rem;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* ── KPI stat cards ── */
    .stat-row {
      display: flex; flex-wrap: wrap; gap: 1rem;
      max-width: 1280px; margin: 0 auto 2.5rem;
    }
    .stat-card {
      flex: 1 1 180px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.2rem 1.4rem;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .stat-value { font-size: 2rem; font-weight: bold; line-height: 1.1; }
    .stat-label { font-size: 0.85rem; text-transform: uppercase;
                  letter-spacing: 0.1em; color: var(--muted); margin-top: 0.35rem; }
    .stat-sub   { font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem;
                  font-style: italic; }

    /* ── Image gallery ── */
    .gallery { display: flex; flex-wrap: wrap; gap: 1rem; }
    .gallery-item { flex: 1 1 300px; }
    .gallery-item img { width: 100%; border-radius: 4px; display: block; }
    .gallery-item figcaption {
      text-align: center; font-size: 0.8rem;
      color: var(--muted); margin-top: 0.4rem; font-style: italic;
    }

    /* ── Single prediction image ── */
    .img-section img {
      width: 100%; max-width: 1100px;
      display: block; margin: 0 auto;
      border-radius: 4px;
    }

    /* ── Footer ── */
    footer {
      max-width: 1280px; margin: 0 auto;
      font-size: 0.8rem; color: var(--muted);
      text-align: center; font-style: italic;
    }
    
    /* ── Section headings ── */
    .section-heading {
      text-align: center;
      margin: 0 0 1.8rem;
    }
    .section-heading h2 {
      font-family: 'Georgia', serif;
      font-size: 1.35rem;
      font-weight: normal;
      letter-spacing: 0.04em;
      color: var(--accent);
      margin: 0 0 0.5rem;
    }
    .section-rule {
      width: 60%;
      height: 1px;
      background: var(--accent);
      margin: 0 auto;
    }
  </style>
</head>
"""

    html_header = """\
<body>
<header>
  <h1>COVID-19 dynamics in Singapore</h1>
  <p>Authors: Karol Chądzyński, Mykyta Khrabust</p>
</header>
<div class="section-heading">
  <h2>Summary Statistics</h2>
  <div class="section-rule"></div>
</div>
"""

    html_stats = f"{stat_cards_html}\n" if stat_cards_html else ""

    html_charts = f"""\
    
<div class="section">
  <h2>Cases, Deaths &amp; Vaccinations</h2>
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
<div class="section-heading">
  <h2>Inferential Statistics</h2>
  <div class="section-rule"></div>
</div>
<div class="section">
  <h2>New Confirmed Cases — Time Series Prediction</h2>
  <div class="card img-section">
    <img src={cases_src} alt="New confirmed cases prediction"/>
  </div>
</div>

<div class="section">
  <h2>Deceased — Time Series Prediction</h2>
  <div class="card img-section">
    <img src={deaths_src} alt="Deceased time series prediction"/>
  </div>
</div>
"""


    html_footer = """\
<footer>DAV Final project, group 8.</footer>
</body>
</html>"""

    # ── 4. Concatenate and write ──────────────────────────────────────────────
    full_html = (
        html_head
        + html_styles
        + html_header
        + html_stats
        + html_charts
        + html_footer
    )

    with open(figname, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Report saved → {figname}")

# data = pd.read_csv('../data_processed/SG_nona.csv')
# data['date'] = pd.to_datetime(data['date'])
#
# colors = {
#     'CONFIRMED': '#52A929',
#     'VACCINATED': '#00D5D2',
#     'FULLY_VACCINATED': '#D500DA',
#     'DECEASED': '#D50000',
#     'CFR': '#D32F2F',
# }
#
# sg_events = {
#     "Circuit Breaker": "2020-04-07",
#     "Vaccination Starts": "2020-12-30",
#     "Delta Wave": "2021-08-01",
#     "Omicron Wave": "2021-12-15",
# }
#
# build_report(data, color_dict=colors, event_dict=sg_events, figname="report.html")