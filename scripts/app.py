"""
Dash-based interactive forecast visualization application.

Wraps a :class:`Predictor` instance in a self-contained Dash app.
Columns to forecast are selected at runtime via an in-app searchable
dropdown; the constructor requires only a DataFrame.

Usage
-----
    from app import ForecastApp
    import pandas as pd

    df  = pd.read_csv("data/processed/singapore_full.csv", parse_dates=["date"])
    app = ForecastApp(df)
    app.run()

Dependencies
------------
    pip install dash dash-bootstrap-components plotly prophet pmdarima pandas numpy
"""

from __future__ import annotations

import os
import signal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, ctx
import dash_bootstrap_components as dbc

from scripts.predictor import Predictor


_COLORS: dict[str, str] = {
    "train":        "#2c7bb6",
    "actual":       "#333333",
    "prophet_line": "#d7191c",
    "prophet_ci":   "rgba(215, 25, 28, 0.15)",
    "arima_line":   "#1a9641",
    "arima_ci":     "rgba(26, 150, 65, 0.15)",
}

_MIN_GAP: int = 10


class ForecastApp:
    """
    Interactive Dash application for Prophet vs ARIMA forecast exploration.

    Target columns are chosen at runtime through a searchable multi-select
    dropdown inside the app — no columns need to be declared at construction.
    The layout provides three sliders (``date_min``, ``date_max``,
    ``date_pred``), mutual slider clamping to enforce a minimum gap of
    ``_MIN_GAP`` days between ``date_min`` and ``date_max``, human-readable
    date tooltips on the date sliders, quarterly tick marks that never
    overlap, a **Calculate** button that triggers model fitting, and a
    **Quit** button that shuts down the server.

    :param df: Preprocessed DataFrame containing a ``date`` column
               (datetime) and at least one numeric column.
               Must have a default RangeIndex.
    :param default_date_min: Initial index for the training-start slider.
               Defaults to ``0``.
    :param default_date_max: Initial index for the training-end slider.
               Defaults to ``min(730, len(df) - 1)``.
    :param default_date_pred: Initial forecast horizon in days.
               Defaults to ``180``.
    :param port: TCP port for the Dash development server. Defaults to ``8050``.
    :param debug: Whether to start Dash in debug mode. Defaults to ``True``.
    :raises TypeError: If ``df`` is not a pandas DataFrame.
    :raises ValueError: If ``df`` lacks a ``date`` column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        default_date_min: int = 0,
        default_date_max: int | None = None,
        default_date_pred: int = 180,
        port: int = 8050,
        debug: bool = True,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"df must be a pandas DataFrame, got {type(df).__name__}."
            )
        if "date" not in df.columns:
            raise ValueError("df must contain a 'date' column.")

        self._df: pd.DataFrame = df.copy()
        self._df["date"] = pd.to_datetime(self._df["date"])
        self._df = self._df.sort_values("date").reset_index(drop=True)
        self._n: int = len(self._df)

        self._default_min:  int = default_date_min
        self._default_max:  int = (
            default_date_max if default_date_max is not None
            else min(730, self._n - 1)
        )
        self._default_pred: int = default_date_pred
        self._port:  int  = port
        self._debug: bool = debug

        self._numeric_cols: list[str] = [
            c for c in self._df.columns
            if c != "date" and pd.api.types.is_numeric_dtype(self._df[c])
        ]

        self._app: Dash = self._build_app()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _idx_to_label(self, idx: int) -> str:
        """
        Format a DataFrame row index as a human-readable date string.

        :param idx: Row index into ``self._df``.
        :return: Date formatted as ``DD Mon YYYY``.
        """
        return self._df["date"].iloc[idx].strftime("%d %b %Y")

    def _quarterly_marks(self) -> dict[int, dict]:
        """
        Build slider tick marks at quarterly boundaries.

        Each mark uses a rotated label style to prevent overlap.
        The first and last indices are always included.

        :return: Dict of ``{index: {"label": str, "style": dict}}`` suitable
                 for the Dash ``dcc.Slider`` ``marks`` prop.
        """
        dates  = self._df["date"]
        marks: dict[int, dict] = {}
        style  = {"transform": "rotate(-40deg)", "white-space": "nowrap",
                  "font-size": "11px"}

        seen_quarters: set[tuple[int, int]] = set()
        for i, d in enumerate(dates):
            key = (d.year, (d.month - 1) // 3)
            if key not in seen_quarters:
                seen_quarters.add(key)
                marks[i] = {
                    "label": d.strftime("Q%q %Y").replace(
                        "Q1", "Q1").replace("Q2", "Q2")
                    .replace("Q3", "Q3").replace("Q4", "Q4"),
                    "style": style,
                }

        # strftime %q not portable — compute quarter manually
        marks = {}
        seen_quarters = set()
        for i, d in enumerate(dates):
            q   = (d.month - 1) // 3 + 1
            key = (d.year, q)
            if key not in seen_quarters:
                seen_quarters.add(key)
                marks[i] = {
                    "label": f"Q{q} {d.year}",
                    "style": style,
                }

        marks[0]          = {"label": dates.iloc[0].strftime("%d %b %Y"),  "style": style}
        marks[self._n -1] = {"label": dates.iloc[-1].strftime("%d %b %Y"), "style": style}
        return marks

    @staticmethod
    def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Compute Root Mean Squared Error ignoring NaN pairs.

        :param actual: Array of observed values.
        :param predicted: Array of predicted values.
        :return: RMSE scalar, or ``nan`` if no valid pairs exist.
        """
        mask = ~np.isnan(actual) & ~np.isnan(predicted)
        if mask.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))

    def _empty_figure(self) -> go.Figure:
        """
        Return a blank placeholder figure.

        :return: Empty :class:`go.Figure` with white background.
        """
        return go.Figure().update_layout(
            plot_bgcolor="white", paper_bgcolor="white", height=420,
        )

    def _make_figure(
        self,
        col: str,
        train: pd.DataFrame,
        actual_post: pd.DataFrame | None,
        prophet_fc: pd.DataFrame,
        arima_fc: pd.DataFrame,
    ) -> go.Figure:
        """
        Build a Plotly figure for one target column.

        :param col: Target column name.
        :param train: Training slice of the DataFrame.
        :param actual_post: Rows after ``date_max`` (held-out actuals), or
               ``None`` if the training window reaches the end of the data.
        :param prophet_fc: Prophet forecast DataFrame.
        :param arima_fc: ARIMA forecast DataFrame.
        :return: Populated :class:`go.Figure`.
        """
        yhat_col  = f"{col}_yhat"
        lower_col = f"{col}_yhat_lower"
        upper_col = f"{col}_yhat_upper"

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=train["date"], y=train[col],
            mode="lines", name="Training data",
            line={"color": _COLORS["train"], "width": 1.5},
        ))

        if actual_post is not None and len(actual_post) > 0:
            fig.add_trace(go.Scatter(
                x=actual_post["date"], y=actual_post[col],
                mode="lines", name="Actual (post training)",
                line={"color": _COLORS["actual"], "width": 1.5, "dash": "dot"},
            ))

        for fc, lc, cc, name in (
            (prophet_fc, _COLORS["prophet_line"], _COLORS["prophet_ci"], "Prophet"),
            (arima_fc,   _COLORS["arima_line"],   _COLORS["arima_ci"],   "ARIMA"),
        ):
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["date"], fc["date"].iloc[::-1]]),
                y=pd.concat([fc[upper_col], fc[lower_col].iloc[::-1]]),
                fill="toself", fillcolor=cc,
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=fc["date"], y=fc[yhat_col],
                mode="lines", name=f"{name} forecast",
                line={"color": lc, "width": 2, "dash": "dash"},
            ))

        fig.add_vline(
            x=train["date"].iloc[-1].timestamp() * 1000,
            line_dash="dash", line_color="grey", line_width=1,
            annotation_text="forecast start",
            annotation_position="top right",
            annotation_font_size=10,
        )
        fig.update_layout(
            title={"text": col, "font": {"size": 15}},
            xaxis={"title": "Date", "showgrid": False},
            yaxis={"title": col, "tickformat": ",.0f",
                   "showgrid": True, "gridcolor": "#eeeeee"},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                    "xanchor": "right", "x": 1},
            hovermode="x unified",
            plot_bgcolor="white", paper_bgcolor="white",
            margin={"t": 60, "b": 40, "l": 60, "r": 20},
            height=420,
        )
        return fig

    def _compute_badges(
        self,
        columns: list[str],
        actual_post: pd.DataFrame | None,
        prophet_fc: pd.DataFrame,
        arima_fc: pd.DataFrame,
    ) -> list:
        """
        Build RMSE metric badges for the given target columns.

        :param columns: Column names that were fitted.
        :param actual_post: Held-out rows after ``date_max``, or ``None``.
        :param prophet_fc: Prophet forecast DataFrame.
        :param arima_fc: ARIMA forecast DataFrame.
        :return: List of :class:`dbc.Badge` components, possibly empty.
        """
        if actual_post is None or len(actual_post) == 0:
            return []
        badges: list = []
        for col in columns:
            yhat_col = f"{col}_yhat"
            merged = (
                actual_post[["date", col]]
                .merge(prophet_fc[["date", yhat_col]].rename(
                    columns={yhat_col: "prophet"}), on="date", how="inner")
                .merge(arima_fc[["date", yhat_col]].rename(
                    columns={yhat_col: "arima"}), on="date", how="inner")
            )
            if len(merged) == 0:
                continue
            p_rmse = self._rmse(merged[col].values, merged["prophet"].values)
            a_rmse = self._rmse(merged[col].values, merged["arima"].values)
            badges += [
                dbc.Badge(f"{col} · Prophet RMSE: {p_rmse:,.0f}",
                          color="danger", className="me-1"),
                dbc.Badge(f"{col} · ARIMA RMSE: {a_rmse:,.0f}",
                          color="success", className="me-1"),
            ]
        return badges

    # ------------------------------------------------------------------
    # App construction
    # ------------------------------------------------------------------

    def _build_app(self) -> Dash:
        """
        Construct and wire the Dash application.

        :return: Configured :class:`Dash` application object.
        """
        app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.FLATLY],
            title="COVID-19 Forecast Explorer",
        )

        marks = self._quarterly_marks()
        dropdown_options = [{"label": c, "value": c} for c in self._numeric_cols]

        app.layout = dbc.Container(
            fluid=True,
            style={"maxWidth": "1600px", "padding": "2rem"},
            children=[

                dbc.Row(dbc.Col(html.H2(
                    "COVID-19 Singapore — Prophet vs ARIMA Forecast Explorer",
                    className="text-center mb-1",
                ))),
                dbc.Row(dbc.Col(html.P(
                    "Select columns, adjust the sliders, then click Calculate.",
                    className="text-center text-muted mb-4",
                ))),

                # ── Column selector ──────────────────────────────────────
                dbc.Card(dbc.CardBody([
                    html.Label("Columns to forecast", className="fw-bold mb-1"),
                    dcc.Dropdown(
                        id="dropdown-columns",
                        options=dropdown_options,
                        value=self._numeric_cols[:2] if len(self._numeric_cols) >= 2
                              else self._numeric_cols[:1],
                        multi=True,
                        placeholder="Search and select columns…",
                        style={"font-size": "13px"},
                    ),
                ]), className="mb-3 shadow-sm"),

                # ── Sliders ──────────────────────────────────────────────
                dbc.Card(dbc.CardBody([
                    dbc.Row([

                        dbc.Col([
                            html.Label(
                                id="label-date-min",
                                className="fw-bold",
                            ),
                            dcc.Slider(
                                id="slider-date-min",
                                min=0,
                                max=self._n - 2,
                                step=1,
                                value=self._default_min,
                                marks=marks,
                                tooltip={"placement": "bottom",
                                         "always_visible": False},
                                updatemode="drag",
                            ),
                        ], md=4),

                        dbc.Col([
                            html.Label(
                                id="label-date-max",
                                className="fw-bold",
                            ),
                            dcc.Slider(
                                id="slider-date-max",
                                min=1,
                                max=self._n - 1,
                                step=1,
                                value=self._default_max,
                                marks=marks,
                                tooltip={"placement": "bottom",
                                         "always_visible": False},
                                updatemode="drag",
                            ),
                        ], md=4),

                        dbc.Col([
                            html.Label(
                                id="label-date-pred",
                                className="fw-bold",
                            ),
                            dcc.Slider(
                                id="slider-date-pred",
                                min=7, max=730, step=7,
                                value=self._default_pred,
                                marks={
                                    7: {"label": "1 wk"},
                                    30: {"label": "1 mo"},
                                    90: {"label": "3 mo"},
                                    180: {"label": "6 mo"},
                                    365: {"label": "1 yr"},
                                    730: {"label": "2 yr"},
                                },
                                tooltip={"placement": "bottom",
                                         "always_visible": False},
                                updatemode="drag",
                            ),
                        ], md=4),

                    ]),

                    dbc.Row(dbc.Col(
                        dbc.Alert(id="alert-validation",
                                  color="danger", is_open=False),
                        md=12, className="mt-2",
                    )),

                    dbc.Row([
                        dbc.Col(dbc.Button(
                            "Calculate", id="btn-calculate",
                            color="primary", n_clicks=0,
                            className="w-100",
                        ), md=2),
                        dbc.Col(dbc.Button(
                            "Quit", id="btn-quit",
                            color="danger", outline=True,
                            n_clicks=0, className="w-100",
                        ), md=2),
                    ], className="mt-3"),

                ]), className="mb-4 shadow-sm"),

                # ── Status + badges ──────────────────────────────────────
                dbc.Row([
                    dbc.Col(dbc.Spinner(
                        html.Div(id="status-text",
                                 className="text-muted small"),
                        size="sm", color="primary",
                    ), md=6),
                    dbc.Col(html.Div(id="metric-badges"),
                            md=6, className="text-end"),
                ], className="mb-3"),

                # ── Plot container (filled dynamically) ──────────────────
                html.Div(id="graph-container"),

                dbc.Row(dbc.Col(html.P(
                    "Data: Google Open COVID-19 Dataset · "
                    "Models: Prophet (Meta), ARIMA (pmdarima)",
                    className="text-center text-muted small mt-2",
                ))),
            ],
        )

        # ------------------------------------------------------------------
        # Callback: mutual slider clamping + live labels
        # ------------------------------------------------------------------

        @app.callback(
            Output("slider-date-min",  "value"),
            Output("slider-date-max",  "value"),
            Output("label-date-min",   "children"),
            Output("label-date-max",   "children"),
            Output("label-date-pred",  "children"),
            Input("slider-date-min",   "value"),
            Input("slider-date-max",   "value"),
            Input("slider-date-pred",  "value"),
        )
        def _clamp_sliders(
            date_min: int,
            date_max: int,
            date_pred: int,
        ) -> tuple[int, int, str, str, str]:
            """
            Enforce a minimum gap between ``date_min`` and ``date_max`` and
            update all slider labels with human-readable values.

            When ``date_min`` is moved too close to (or past) ``date_max``,
            ``date_max`` is pushed forward.  The symmetric case pushes
            ``date_min`` backward.  Neither slider is allowed to go out of
            the ``[0, n-1]`` range.

            :param date_min: Current value of the training-start slider.
            :param date_max: Current value of the training-end slider.
            :param date_pred: Current value of the forecast-horizon slider.
            :return: Clamped ``date_min``, clamped ``date_max``, and three
                     label strings for the slider headers.
            """
            triggered = ctx.triggered_id

            if triggered == "slider-date-min":
                if date_min >= date_max - _MIN_GAP:
                    date_max = min(date_min + _MIN_GAP, self._n - 1)
            elif triggered == "slider-date-max":
                if date_max <= date_min + _MIN_GAP:
                    date_min = max(date_max - _MIN_GAP, 0)

            label_min  = f"Training start — {self._idx_to_label(date_min)}"
            label_max  = f"Training end — {self._idx_to_label(date_max)}"
            label_pred = f"Forecast horizon — {date_pred} days"

            return date_min, date_max, label_min, label_max, label_pred

        # ------------------------------------------------------------------
        # Callback: fit models and render plots
        # ------------------------------------------------------------------

        @app.callback(
            Output("graph-container",  "children"),
            Output("metric-badges",    "children"),
            Output("status-text",      "children"),
            Output("alert-validation", "children"),
            Output("alert-validation", "is_open"),
            Input("btn-calculate",     "n_clicks"),
            State("slider-date-min",   "value"),
            State("slider-date-max",   "value"),
            State("slider-date-pred",  "value"),
            State("dropdown-columns",  "value"),
            prevent_initial_call=True,
        )
        def _update(
            _n_clicks: int,
            date_min: int,
            date_max: int,
            date_pred: int,
            columns: list[str] | None,
        ) -> tuple:
            """
            Re-fit Predictor for the selected columns and render one figure
            per column when Calculate is clicked.

            :param _n_clicks: Click counter (unused beyond triggering).
            :param date_min: Training window start index.
            :param date_max: Training window end index.
            :param date_pred: Forecast horizon in days.
            :param columns: Column names chosen in the dropdown.
            :return: Tuple of graph container children, badges, status text,
                     alert message, and alert visibility flag.
            """
            if not columns:
                msg = "Please select at least one column."
                return [], [], msg, msg, True

            if date_min >= date_max:
                msg = (
                    f"date_min ({date_min}) must be strictly less than "
                    f"date_max ({date_max})."
                )
                return [], [], msg, msg, True

            status = (
                f"Training: {self._idx_to_label(date_min)} → "
                f"{self._idx_to_label(date_max)} "
                f"({date_max - date_min} days) · "
                f"Forecasting {date_pred} days ahead"
            )

            try:
                predictor = Predictor(self._df, columns=columns)
                predictor.fit(date_min, date_max, date_pred)
                forecasts = predictor.get_forecast()
            except Exception as exc:
                return [], [], str(exc), str(exc), True

            prophet_fc  = forecasts["prophet"]
            arima_fc    = forecasts["arima"]
            train       = self._df.iloc[date_min: date_max + 1]
            actual_post = (
                self._df.iloc[date_max + 1:]
                if date_max + 1 < self._n else None
            )

            graphs = [
                dbc.Row(
                    dbc.Col(dcc.Graph(
                        figure=self._make_figure(
                            col, train, actual_post, prophet_fc, arima_fc,
                        ),
                        config={"displayModeBar": True},
                    ), md=12),
                    className="mb-3",
                )
                for col in columns
            ]
            badges = self._compute_badges(
                columns, actual_post, prophet_fc, arima_fc,
            )
            return graphs, badges, status, "", False

        # ------------------------------------------------------------------
        # Callback: quit
        # ------------------------------------------------------------------

        @app.callback(
            Output("btn-quit", "disabled"),
            Input("btn-quit",  "n_clicks"),
            prevent_initial_call=True,
        )
        def _quit(_n_clicks: int) -> bool:
            """
            Shut down the server on Quit button click.

            :param _n_clicks: Click counter (unused beyond triggering).
            :return: ``True`` to disable the button before the process exits.
            """
            os.kill(os.getpid(), signal.SIGTERM)
            return True

        return app

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the Dash development server.

        Blocks until the server is stopped (Ctrl-C or Quit button).
        Opens at ``http://127.0.0.1:<port>``.
        """
        self._app.run(debug=self._debug, port=self._port)
