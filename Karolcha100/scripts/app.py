"""
Dash-based interactive forecast visualization application.

Wraps a :class:`Predictor` instance in a self-contained Dash app that
exposes sliders for ``date_min``, ``date_max``, and ``date_pred`` and
re-fits both Prophet and ARIMA models on every slider interaction.

Usage
-----
    from app import ForecastApp
    import pandas as pd

    df  = pd.read_csv("data/processed/singapore_full.csv", parse_dates=["date"])
    app = ForecastApp(df, columns=["new_confirmed_7d", "new_deceased_7d"])
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
from dash import Dash, Input, Output, State, dcc, html
import dash_bootstrap_components as dbc

from predictor import Predictor


# ---------------------------------------------------------------------------
# Module-level colour palette (not user-configurable — presentational only)
# ---------------------------------------------------------------------------

_COLORS: dict[str, str] = {
    "train":       "#2c7bb6",
    "actual":      "#333333",
    "prophet_line": "#d7191c",
    "prophet_ci":  "rgba(215, 25, 28, 0.15)",
    "arima_line":  "#1a9641",
    "arima_ci":    "rgba(26, 150, 65, 0.15)",
}


class ForecastApp:
    """
    Interactive Dash application for Prophet vs ARIMA forecast exploration.

    Accepts a preprocessed DataFrame and a list of numeric target columns.
    Builds a Dash layout with three sliders (``date_min``, ``date_max``,
    ``date_pred``), a **Calculate** button that triggers model fitting, a
    **Quit** button that shuts down the server, and one Plotly figure per
    target column.  Models are fitted only when Calculate is clicked, not
    on every slider move.

    :param df: Preprocessed DataFrame containing a ``date`` column
               (datetime) and all columns listed in ``columns``.
               Must have a default RangeIndex.
    :param columns: Target column names to forecast and visualise.
               Each column gets its own plot panel.
    :param column_labels: Optional mapping ``{column_name: display_label}``
               used as axis titles and panel headings.  Columns absent from
               this dict fall back to the raw column name.
    :param default_date_min: Initial value of the training-start slider.
               Defaults to ``0``.
    :param default_date_max: Initial value of the training-end slider.
               Defaults to ``min(730, len(df) - 1)``.
    :param default_date_pred: Initial forecast horizon in days.
               Defaults to ``180``.
    :param port: TCP port for the Dash development server. Defaults to ``8050``.
    :param debug: Whether to start Dash in debug mode. Defaults to ``True``.
    :raises TypeError: If ``df`` is not a pandas DataFrame.
    :raises ValueError: If ``df`` lacks a ``date`` column or any column in
               ``columns`` is absent from ``df``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        columns: list[str],
        column_labels: dict[str, str] | None = None,
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

        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns not found in df: {missing}. "
                f"Available: {list(df.columns)}."
            )

        self._df: pd.DataFrame = df.copy()
        self._df["date"] = pd.to_datetime(self._df["date"])
        self._df = self._df.sort_values("date").reset_index(drop=True)

        self._columns: list[str] = list(columns)
        self._labels: dict[str, str] = column_labels or {}
        self._n: int = len(self._df)

        self._default_min: int = default_date_min
        self._default_max: int = (
            default_date_max
            if default_date_max is not None
            else min(730, self._n - 1)
        )
        self._default_pred: int = default_date_pred
        self._port: int = port
        self._debug: bool = debug

        self._app: Dash = self._build_app()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label(self, col: str) -> str:
        """
        Return the display label for a column, falling back to its name.

        :param col: Column name.
        :return: Human-readable label string.
        """
        return self._labels.get(col, col)

    def _slider_marks(self, step: int = 90) -> dict[int, str]:
        """
        Build Dash slider mark labels at regular date intervals.

        :param step: Interval between labelled marks in days.
        :return: Dict mapping integer index to formatted date string.
        """
        marks: dict[int, str] = {}
        for i in range(0, self._n, step):
            marks[i] = self._df["date"].iloc[i].strftime("%b %Y")
        marks[self._n - 1] = self._df["date"].iloc[-1].strftime("%b %Y")
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
        Return a blank Plotly figure used as a placeholder on errors.

        :return: Empty :class:`go.Figure` with white background.
        """
        return go.Figure().update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=420,
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

        Renders training data, optional held-out actuals, and both model
        forecasts with 95 % confidence bands.

        :param col: Target column name.
        :param train: Training slice of the DataFrame.
        :param actual_post: Rows after ``date_max`` (held-out actuals), or
               ``None`` if the training window reaches the end of the data.
        :param prophet_fc: Prophet forecast DataFrame from
               :meth:`Predictor.get_forecast`.
        :param arima_fc: ARIMA forecast DataFrame from
               :meth:`Predictor.get_forecast`.
        :return: Populated :class:`go.Figure`.
        """
        yhat_col  = f"{col}_yhat"
        lower_col = f"{col}_yhat_lower"
        upper_col = f"{col}_yhat_upper"
        label     = self._label(col)

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

        for fc, line_color, ci_color, name in (
            (prophet_fc, _COLORS["prophet_line"], _COLORS["prophet_ci"], "Prophet"),
            (arima_fc,   _COLORS["arima_line"],   _COLORS["arima_ci"],   "ARIMA"),
        ):
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["date"], fc["date"].iloc[::-1]]),
                y=pd.concat([fc[upper_col], fc[lower_col].iloc[::-1]]),
                fill="toself", fillcolor=ci_color,
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip", showlegend=False,
                name=f"{name} 95% CI",
            ))
            fig.add_trace(go.Scatter(
                x=fc["date"], y=fc[yhat_col],
                mode="lines", name=f"{name} forecast",
                line={"color": line_color, "width": 2, "dash": "dash"},
            ))

        split_date = train["date"].iloc[-1]
        fig.add_vline(
            x=split_date.timestamp() * 1000,
            line_dash="dash", line_color="grey", line_width=1,
            annotation_text="forecast start",
            annotation_position="top right",
            annotation_font_size=10,
        )

        fig.update_layout(
            title={"text": label, "font": {"size": 15}},
            xaxis={"title": "Date", "showgrid": False},
            yaxis={
                "title": label,
                "tickformat": ",.0f",
                "showgrid": True,
                "gridcolor": "#eeeeee",
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom", "y": 1.02,
                "xanchor": "right",  "x": 1,
            },
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin={"t": 60, "b": 40, "l": 60, "r": 20},
            height=420,
        )
        return fig

    def _compute_badges(
        self,
        actual_post: pd.DataFrame | None,
        prophet_fc: pd.DataFrame,
        arima_fc: pd.DataFrame,
    ) -> list:
        """
        Build RMSE metric badges for all target columns.

        Badges are only produced for columns where held-out actuals overlap
        with the forecast period.

        :param actual_post: Held-out rows after ``date_max``, or ``None``.
        :param prophet_fc: Prophet forecast DataFrame.
        :param arima_fc: ARIMA forecast DataFrame.
        :return: List of :class:`dbc.Badge` components, possibly empty.
        """
        if actual_post is None or len(actual_post) == 0:
            return []

        badges: list = []
        for col in self._columns:
            yhat_col = f"{col}_yhat"
            merged = (
                actual_post[["date", col]]
                .merge(
                    prophet_fc[["date", yhat_col]].rename(columns={yhat_col: "prophet"}),
                    on="date", how="inner",
                )
                .merge(
                    arima_fc[["date", yhat_col]].rename(columns={yhat_col: "arima"}),
                    on="date", how="inner",
                )
            )
            if len(merged) == 0:
                continue
            p_rmse = self._rmse(merged[col].values, merged["prophet"].values)
            a_rmse = self._rmse(merged[col].values, merged["arima"].values)
            short  = self._label(col)
            badges += [
                dbc.Badge(
                    f"{short} · Prophet RMSE: {p_rmse:,.0f}",
                    color="danger", className="me-1",
                ),
                dbc.Badge(
                    f"{short} · ARIMA RMSE: {a_rmse:,.0f}",
                    color="success", className="me-1",
                ),
            ]
        return badges

    # ------------------------------------------------------------------
    # App construction
    # ------------------------------------------------------------------

    def _build_app(self) -> Dash:
        """
        Construct and wire the Dash application.

        Builds the layout, registers the single callback that re-fits
        :class:`Predictor` on slider changes, and returns the ready-to-run
        :class:`Dash` instance.

        :return: Configured :class:`Dash` application object.
        """
        app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.FLATLY],
            title="COVID-19 Forecast Explorer",
        )

        marks = self._slider_marks()

        graph_rows = [
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id=f"graph-{col}",
                        config={"displayModeBar": True},
                    ),
                    md=12,
                ),
                className="mb-3",
            )
            for col in self._columns
        ]

        app.layout = dbc.Container(
            fluid=True,
            style={"maxWidth": "1600px", "padding": "2rem"},
            children=[
                dbc.Row(dbc.Col(html.H2(
                    "COVID-19 Singapore — Prophet vs ARIMA Forecast Explorer",
                    className="text-center mb-1",
                ))),
                dbc.Row(dbc.Col(html.P(
                    "Adjust the sliders to define the training window and "
                    "forecast horizon, then click Calculate to fit the models.",
                    className="text-center text-muted mb-4",
                ))),

                dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Training start (date_min)", className="fw-bold"),
                            dcc.Slider(
                                id="slider-date-min",
                                min=0, max=self._n - 2, step=1,
                                value=self._default_min,
                                marks=marks,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], md=4),
                        dbc.Col([
                            html.Label("Training end (date_max)", className="fw-bold"),
                            dcc.Slider(
                                id="slider-date-max",
                                min=1, max=self._n - 1, step=1,
                                value=self._default_max,
                                marks=marks,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], md=4),
                        dbc.Col([
                            html.Label("Forecast horizon — days (date_pred)", className="fw-bold"),
                            dcc.Slider(
                                id="slider-date-pred",
                                min=7, max=730, step=7,
                                value=self._default_pred,
                                marks={
                                    7: "1 wk", 30: "1 mo", 90: "3 mo",
                                    180: "6 mo", 365: "1 yr", 730: "2 yr",
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], md=4),
                    ]),
                    dbc.Row(dbc.Col(
                        dbc.Alert(
                            id="alert-validation",
                            color="danger",
                            is_open=False,
                        ),
                        md=12, className="mt-2",
                    )),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    "Calculate",
                                    id="btn-calculate",
                                    color="primary",
                                    n_clicks=0,
                                    className="w-100",
                                ),
                                md=2,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Quit",
                                    id="btn-quit",
                                    color="danger",
                                    outline=True,
                                    n_clicks=0,
                                    className="w-100",
                                ),
                                md=2,
                            ),
                        ],
                        className="mt-3",
                    ),
                ]), className="mb-4 shadow-sm"),

                dbc.Row([
                    dbc.Col(
                        dbc.Spinner(
                            html.Div(id="status-text", className="text-muted small"),
                            size="sm", color="primary",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        html.Div(id="metric-badges"),
                        md=6, className="text-end",
                    ),
                ], className="mb-3"),

                *graph_rows,

                dbc.Row(dbc.Col(html.P(
                    "Data: Google Open COVID-19 Dataset · "
                    "Models: Prophet (Meta), ARIMA (pmdarima)",
                    className="text-center text-muted small mt-2",
                ))),
            ],
        )

        # Dynamic output list: one figure output per column
        graph_outputs = [
            Output(f"graph-{col}", "figure")
            for col in self._columns
        ]
        all_outputs = [
            *graph_outputs,
            Output("metric-badges",    "children"),
            Output("status-text",      "children"),
            Output("alert-validation", "children"),
            Output("alert-validation", "is_open"),
        ]

        @app.callback(
            all_outputs,
            Input("btn-calculate",    "n_clicks"),
            State("slider-date-min",  "value"),
            State("slider-date-max",  "value"),
            State("slider-date-pred", "value"),
            prevent_initial_call=True,
        )
        def _update(
            _n_clicks: int,
            date_min: int,
            date_max: int,
            date_pred: int,
        ) -> tuple:
            """
            Re-fit Predictor and rebuild all figures when Calculate is clicked.

            Sliders are read as ``State`` so the callback fires only on button
            click, not on every slider movement.

            :param _n_clicks: Click counter from the Calculate button (unused
                   beyond triggering the callback).
            :param date_min: Training window start index from slider state.
            :param date_max: Training window end index from slider state.
            :param date_pred: Forecast horizon in days from slider state.
            :return: Tuple of figure objects, badges, status text,
                     alert message, and alert visibility flag.
            """
            n_cols  = len(self._columns)
            empties = [self._empty_figure()] * n_cols

            if date_min >= date_max:
                msg = (
                    f"date_min ({date_min}) must be strictly less "
                    f"than date_max ({date_max})."
                )
                return (*empties, [], msg, msg, True)

            train_start = self._df["date"].iloc[date_min].strftime("%d %b %Y")
            train_end   = self._df["date"].iloc[date_max].strftime("%d %b %Y")
            status = (
                f"Training: {train_start} → {train_end} "
                f"({date_max - date_min} days) · "
                f"Forecasting {date_pred} days ahead"
            )

            try:
                predictor = Predictor(self._df, columns=self._columns)
                predictor.fit(date_min, date_max, date_pred)
                forecasts = predictor.get_forecast()
            except Exception as exc:
                return (*empties, [], str(exc), str(exc), True)

            prophet_fc  = forecasts["prophet"]
            arima_fc    = forecasts["arima"]
            train       = self._df.iloc[date_min: date_max + 1]
            actual_post = (
                self._df.iloc[date_max + 1:]
                if date_max + 1 < self._n
                else None
            )

            figures = [
                self._make_figure(col, train, actual_post, prophet_fc, arima_fc)
                for col in self._columns
            ]
            badges = self._compute_badges(actual_post, prophet_fc, arima_fc)

            return (*figures, badges, status, "", False)

        @app.callback(
            Output("btn-quit", "disabled"),
            Input("btn-quit",  "n_clicks"),
            prevent_initial_call=True,
        )
        def _quit(_n_clicks: int) -> bool:
            """
            Shut down the Dash development server on Quit button click.

            Sends ``SIGTERM`` to the current process, which causes Werkzeug /
            Gunicorn to perform a clean shutdown.  The button is disabled
            immediately to prevent double-clicks.

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

        Blocks until the server is stopped (Ctrl-C).
        Opens at ``http://127.0.0.1:<port>``.
        """
        self._app.run(debug=self._debug, port=self._port)
