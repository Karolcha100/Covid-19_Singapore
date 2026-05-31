"""
Predictor class for time-series forecasting using Prophet and ARIMA.

Supports multiple target columns, index-based train window specification,
and returns per-model forecast DataFrames.

Dependencies
------------
    pip install prophet pmdarima pandas numpy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from prophet import Prophet
from pmdarima import auto_arima


class Predictor:
    """
    Unified forecasting wrapper for Prophet and ARIMA models.

    Target columns are declared at construction time. Training window
    and forecast horizon are passed to :meth:`fit` and can be adjusted
    afterwards via the provided setters.

    :param df: Input DataFrame containing a ``date`` column and numeric
               target columns. Must have a default RangeIndex.
    :param columns: List of column names in ``df`` to forecast.
    :raises TypeError: If ``df`` is not a pandas DataFrame.
    :raises ValueError: If ``df`` lacks a ``date`` column or any requested
                        column is absent from ``df``.
    """

    _MODELS = ("prophet", "arima")

    def __init__(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
        if "date" not in df.columns:
            raise ValueError("df must contain a 'date' column.")

        self._df = df.copy()
        self._df["date"] = pd.to_datetime(self._df["date"])

        self._validate_columns(columns)
        self._columns: list[str] = list(columns)

        self._date_min:  int | None = None
        self._date_max:  int | None = None
        self._date_pred: int | None = None
        self._forecast:  dict[str, pd.DataFrame] | None = None

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_indices(self, date_min: int, date_max: int) -> None:
        """
        Validate that ``date_min`` and ``date_max`` are legal DataFrame index positions.

        :param date_min: Proposed start index.
        :param date_max: Proposed end index.
        :raises IndexError: If either index is out of the DataFrame bounds.
        :raises ValueError: If ``date_min >= date_max``.
        """
        max_idx = len(self._df) - 1
        if not (0 <= date_min <= max_idx):
            raise IndexError(
                f"date_min={date_min} is out of bounds [0, {max_idx}]."
            )
        if not (0 <= date_max <= max_idx):
            raise IndexError(
                f"date_max={date_max} is out of bounds [0, {max_idx}]."
            )
        if date_min >= date_max:
            raise ValueError(
                f"date_min ({date_min}) must be strictly less than date_max ({date_max})."
            )

    def _validate_pred(self, date_pred: int) -> None:
        """
        Validate that ``date_pred`` is a positive integer.

        :param date_pred: Number of forecast steps.
        :raises ValueError: If ``date_pred`` is not positive.
        """
        if date_pred <= 0:
            raise ValueError(
                f"date_pred must be a positive integer, got {date_pred}."
            )

    def _validate_columns(self, columns: list[str]) -> None:
        """
        Validate that all requested target columns exist in the DataFrame.

        :param columns: List of column names to forecast.
        :raises ValueError: If any column is missing from the DataFrame.
        """
        missing = [c for c in columns if c not in self._df.columns]
        if missing:
            raise ValueError(
                f"Columns not found in DataFrame: {missing}. "
                f"Available columns: {list(self._df.columns)}."
            )

    # ------------------------------------------------------------------
    # Getters and setters
    # ------------------------------------------------------------------

    def get_date_min(self) -> int | None:
        """
        Return the current training window start index.

        :return: Start index (inclusive), or ``None`` if not yet set.
        """
        return self._date_min

    def set_date_min(self, date_min: int) -> None:
        """
        Set a new training window start index.

        Invalidates any existing forecast, requiring a new :meth:`fit` call.

        :param date_min: New start index. Must satisfy ``0 <= date_min < date_max``.
        :raises IndexError: If ``date_min`` is out of DataFrame bounds.
        :raises ValueError: If ``date_min >= current date_max``.
        """
        current_max = self._date_max if self._date_max is not None else len(self._df) - 1
        self._validate_indices(date_min, current_max)
        self._date_min = date_min
        self._forecast = None

    def get_date_max(self) -> int | None:
        """
        Return the current training window end index.

        :return: End index (inclusive), or ``None`` if not yet set.
        """
        return self._date_max

    def set_date_max(self, date_max: int) -> None:
        """
        Set a new training window end index.

        Invalidates any existing forecast, requiring a new :meth:`fit` call.

        :param date_max: New end index. Must satisfy ``date_min < date_max <= len(df)-1``.
        :raises IndexError: If ``date_max`` is out of DataFrame bounds.
        :raises ValueError: If ``date_max <= current date_min``.
        """
        current_min = self._date_min if self._date_min is not None else 0
        self._validate_indices(current_min, date_max)
        self._date_max = date_max
        self._forecast = None

    def get_date_pred(self) -> int | None:
        """
        Return the current number of forecast steps.

        :return: Number of days to forecast, or ``None`` if not yet set.
        """
        return self._date_pred

    def set_date_pred(self, date_pred: int) -> None:
        """
        Set a new number of forecast steps.

        Invalidates any existing forecast, requiring a new :meth:`fit` call.

        :param date_pred: Number of future days to forecast. Must be positive.
        :raises ValueError: If ``date_pred`` is not positive.
        """
        self._validate_pred(date_pred)
        self._date_pred = date_pred
        self._forecast = None

    # ------------------------------------------------------------------
    # Internal model helpers
    # ------------------------------------------------------------------

    def _get_train_series(self, column: str) -> pd.DataFrame:
        """
        Slice the training window for a single target column.

        :param column: Target column name.
        :return: DataFrame with ``date`` and ``y`` columns covering
                 rows ``[date_min, date_max]`` (inclusive), NaNs dropped.
        """
        train = self._df.iloc[self._date_min: self._date_max + 1][["date", column]].copy()
        train = train.rename(columns={column: "y"})
        return train.dropna(subset=["y"]).reset_index(drop=True)

    def _future_dates(self, last_date: pd.Timestamp) -> pd.DatetimeIndex:
        """
        Generate a forecast date range starting the day after ``last_date``.

        :param last_date: Last date in the training window.
        :return: DatetimeIndex of length ``date_pred``.
        """
        return pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self._date_pred,
            freq="D",
        )

    def _fit_prophet_single(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit a Prophet model on a single column and return a forecast DataFrame.

        Uses logistic growth with a carrying capacity set to 110 % of the
        observed training maximum to model epidemic saturation.

        :param train: DataFrame with ``date`` and ``y`` columns.
        :return: DataFrame with columns ``date``, ``yhat``, ``yhat_lower``,
                 ``yhat_upper`` of length ``date_pred``.
        """
        cap = train["y"].max() * 1.10
        df_p = pd.DataFrame({
            "ds":    train["date"],
            "y":     train["y"],
            "cap":   cap,
            "floor": 0.0,
        })
        model = Prophet(
            growth="logistic",
            changepoint_prior_scale=0.05,
            seasonality_mode="additive",
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=0.95,
        )
        model.fit(df_p)

        future = pd.DataFrame({
            "ds":    self._future_dates(train["date"].iloc[-1]),
            "cap":   cap,
            "floor": 0.0,
        })
        fc = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        fc["yhat"]       = fc["yhat"].clip(lower=0)
        fc["yhat_lower"] = fc["yhat_lower"].clip(lower=0)
        return fc.rename(columns={"ds": "date"}).reset_index(drop=True)

    def _fit_arima_single(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Auto-select and fit an ARIMA model on a single column.

        Uses ``pmdarima.auto_arima`` with seasonal period ``m=7`` to capture
        weekly reporting patterns common in COVID-19 surveillance data.

        :param train: DataFrame with ``date`` and ``y`` columns.
        :return: DataFrame with columns ``date``, ``yhat``, ``yhat_lower``,
                 ``yhat_upper`` of length ``date_pred``.
        """
        model = auto_arima(
            train["y"].values,
            seasonal=True,
            m=7,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
            max_p=3, max_q=3,
            max_P=2, max_Q=2,
        )
        print(f"[ARIMA] Selected order: {model.order}, seasonal: {model.seasonal_order}")
        fc_vals, conf_int = model.predict(
            n_periods=self._date_pred,
            return_conf_int=True,
        )
        return pd.DataFrame({
            "date":       self._future_dates(train["date"].iloc[-1]),
            "yhat":       np.clip(fc_vals, 0, None),
            "yhat_lower": np.clip(conf_int[:, 0], 0, None),
            "yhat_upper": conf_int[:, 1],
        }).reset_index(drop=True)

    @staticmethod
    def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge a list of per-column forecast DataFrames on the ``date`` column.

        :param frames: List of DataFrames each sharing a ``date`` column.
        :return: Single wide DataFrame joined on ``date``.
        """
        result = frames[0]
        for frame in frames[1:]:
            result = result.merge(frame, on="date", how="outer")
        return result.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        date_min: int,
        date_max: int,
        date_pred: int,
    ) -> None:
        """
        Fit Prophet and ARIMA models for all columns declared at construction.

        Validates and stores ``date_min``, ``date_max``, and ``date_pred``,
        then trains both models on rows ``[date_min, date_max]`` of the
        DataFrame and stores forecasts of length ``date_pred`` internally.

        :param date_min: Inclusive start index of the training window.
        :param date_max: Inclusive end index of the training window.
        :param date_pred: Number of future days to forecast. Must be positive.
        :raises IndexError: If ``date_min`` or ``date_max`` are out of bounds.
        :raises ValueError: If ``date_min >= date_max`` or ``date_pred <= 0``.
        """
        self._validate_indices(date_min, date_max)
        self._validate_pred(date_pred)

        self._date_min  = date_min
        self._date_max  = date_max
        self._date_pred = date_pred
        self._forecast  = None

        prophet_frames: list[pd.DataFrame] = []
        arima_frames:   list[pd.DataFrame] = []

        for col in self._columns:
            train = self._get_train_series(col)

            print(f"[Prophet] Fitting column: {col}")
            p_fc = self._fit_prophet_single(train).rename(columns={
                "yhat":       f"{col}_yhat",
                "yhat_lower": f"{col}_yhat_lower",
                "yhat_upper": f"{col}_yhat_upper",
            })
            prophet_frames.append(p_fc)

            print(f"[ARIMA]   Fitting column: {col}")
            a_fc = self._fit_arima_single(train).rename(columns={
                "yhat":       f"{col}_yhat",
                "yhat_lower": f"{col}_yhat_lower",
                "yhat_upper": f"{col}_yhat_upper",
            })
            arima_frames.append(a_fc)

        self._forecast = {
            "prophet": self._merge_frames(prophet_frames),
            "arima":   self._merge_frames(arima_frames),
        }
        print("[OK] Fitting complete.")

    def get_forecast(self) -> dict[str, pd.DataFrame]:
        """
        Return forecast DataFrames keyed by model name.

        Each DataFrame contains a ``date`` column followed by triplets
        ``{col}_yhat``, ``{col}_yhat_lower``, ``{col}_yhat_upper`` for
        every target column declared at construction.

        :return: Dictionary with keys ``"prophet"`` and ``"arima"``, each
                 mapping to a forecast DataFrame of length ``date_pred``.
        :raises RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self._forecast is None:
            raise RuntimeError("No forecast available. Call fit() first.")
        return self._forecast
