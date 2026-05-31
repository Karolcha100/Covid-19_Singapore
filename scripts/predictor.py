import warnings
warnings.filterwarnings("ignore")

from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from pmdarima import auto_arima


class Predictor:
    """
    Unified forecasting wrapper for Prophet and ARIMA models.

    Accepts a DataFrame with a ``date`` column and integer index positions
    defining the training window. Forecasts one or more target columns
    simultaneously using both models.

    :param df: Input DataFrame containing a ``date`` column and numeric
               target columns. Must have a default RangeIndex.
    :param date_min: Inclusive start index of the training window.
    :param date_max: Inclusive end index of the training window.
    :param date_pred: Number of future days to forecast. Must be positive.
    :raises TypeError: If ``df`` is not a pandas DataFrame.
    :raises ValueError: If ``date_pred`` is not positive or ``date_min >= date_max``.
    :raises IndexError: If ``date_min`` or ``date_max`` are outside the DataFrame index.
    """

    _MODELS = ("prophet", "arima")

    def __init__(
        self,
        df: pd.DataFrame,
        date_min: int,
        date_max: int,
        date_pred: int,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
        if "date" not in df.columns:
            raise ValueError("df must contain a 'date' column.")

        self._df = df.copy()
        self._df["date"] = pd.to_datetime(self._df["date"])

        self._forecast: dict[str, pd.DataFrame] | None = None
        self._columns: list[str] | None = None

        self._validate_indices(date_min, date_max)
        self._validate_pred(date_pred)

        self._date_min = date_min
        self._date_max = date_max
        self._date_pred = date_pred

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_indices(self, date_min: int, date_max: int) -> None:
        """
        Validate that date_min and date_max are legal DataFrame index positions.

        :param date_min: Proposed start index.
        :param date_max: Proposed end index.
        :raises IndexError: If either index is out of the DataFrame bounds.
        :raises ValueError: If date_min >= date_max.
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
        Validate that date_pred is a positive integer.

        :param date_pred: Number of forecast steps.
        :raises ValueError: If date_pred is not positive.
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

    def get_date_min(self) -> int:
        """
        Return the current training window start index.

        :return: Start index (inclusive).
        """
        return self._date_min

    def set_date_min(self, date_min: int) -> None:
        """
        Set a new training window start index.

        :param date_min: New start index. Must satisfy 0 <= date_min < date_max.
        :raises IndexError: If date_min is out of DataFrame bounds.
        :raises ValueError: If date_min >= current date_max.
        """
        self._validate_indices(date_min, self._date_max)
        self._date_min = date_min
        self._forecast = None

    def get_date_max(self) -> int:
        """
        Return the current training window end index.

        :return: End index (inclusive).
        """
        return self._date_max

    def set_date_max(self, date_max: int) -> None:
        """
        Set a new training window end index.

        :param date_max: New end index. Must satisfy date_min < date_max <= len(df)-1.
        :raises IndexError: If date_max is out of DataFrame bounds.
        :raises ValueError: If date_max <= current date_min.
        """
        self._validate_indices(self._date_min, date_max)
        self._date_max = date_max
        self._forecast = None

    def get_date_pred(self) -> int:
        """
        Return the current number of forecast steps.

        :return: Number of days to forecast.
        """
        return self._date_pred

    def set_date_pred(self, date_pred: int) -> None:
        """
        Set a new number of forecast steps.

        :param date_pred: Number of future days to forecast. Must be positive.
        :raises ValueError: If date_pred is not positive.
        """
        self._validate_pred(date_pred)
        self._date_pred = date_pred
        self._forecast = None

    # ------------------------------------------------------------------
    # Internal model fitting
    # ------------------------------------------------------------------

    def _get_train_series(self, column: str) -> pd.DataFrame:
        """
        Slice the training window for a single column.

        :param column: Target column name.
        :return: DataFrame with ``date`` and ``y`` columns for the training window.
        """
        train = self._df.iloc[self._date_min: self._date_max + 1][["date", column]].copy()
        train = train.rename(columns={column: "y"})
        train = train.dropna(subset=["y"])
        return train

    def _future_dates(self, last_date: pd.Timestamp) -> pd.DatetimeIndex:
        """
        Generate forecast date range starting the day after last_date.

        :param last_date: Last date in the training window.
        :return: DatetimeIndex of length date_pred.
        """
        return pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self._date_pred,
            freq="D",
        )

    def _fit_prophet_single(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit Prophet on a single column and return a forecast DataFrame.

        :param train: DataFrame with ``date`` and ``y`` columns.
        :return: DataFrame with columns ``date``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
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

        future_dates = self._future_dates(train["date"].iloc[-1])
        future = pd.DataFrame({
            "ds":    future_dates,
            "cap":   cap,
            "floor": 0.0,
        })
        fc = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        fc["yhat"]       = fc["yhat"].clip(lower=0)
        fc["yhat_lower"] = fc["yhat_lower"].clip(lower=0)
        fc = fc.rename(columns={"ds": "date"})
        return fc.reset_index(drop=True)

    def _fit_arima_single(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Auto-select and fit ARIMA on a single column, return a forecast DataFrame.

        :param train: DataFrame with ``date`` and ``y`` columns.
        :return: DataFrame with columns ``date``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
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
        fc_vals, conf_int = model.predict(
            n_periods=self._date_pred,
            return_conf_int=True,
        )
        future_dates = self._future_dates(train["date"].iloc[-1])
        return pd.DataFrame({
            "date":       future_dates,
            "yhat":       np.clip(fc_vals, 0, None),
            "yhat_lower": np.clip(conf_int[:, 0], 0, None),
            "yhat_upper": conf_int[:, 1],
        }).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, columns: list[str]) -> None:
        """
        Fit Prophet and ARIMA models for each requested target column.

        Uses the training window defined by ``date_min`` and ``date_max``.
        Results are stored internally and accessible via ``get_forecast()``.

        :param columns: List of DataFrame column names to forecast.
        :raises ValueError: If any column is not present in the DataFrame.
        """
        self._validate_columns(columns)
        self._columns = columns

        prophet_frames: list[pd.DataFrame] = []
        arima_frames:   list[pd.DataFrame] = []

        for col in columns:
            print(f"[Prophet] Fitting column: {col}")
            train = self._get_train_series(col)
            p_fc = self._fit_prophet_single(train)
            p_fc = p_fc.rename(columns={
                "yhat":       f"{col}_yhat",
                "yhat_lower": f"{col}_yhat_lower",
                "yhat_upper": f"{col}_yhat_upper",
            })
            prophet_frames.append(p_fc)

            print(f"[ARIMA]   Fitting column: {col}")
            a_fc = self._fit_arima_single(train)
            a_fc = a_fc.rename(columns={
                "yhat":       f"{col}_yhat",
                "yhat_lower": f"{col}_yhat_lower",
                "yhat_upper": f"{col}_yhat_upper",
            })
            arima_frames.append(a_fc)

        def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
            result = frames[0]
            for frame in frames[1:]:
                result = result.merge(frame, on="date", how="outer")
            return result.sort_values("date").reset_index(drop=True)

        self._forecast = {
            "prophet": _merge_frames(prophet_frames),
            "arima":   _merge_frames(arima_frames),
        }
        print("[OK] Fitting complete.")

    def get_forecast(self) -> dict[str, pd.DataFrame]:
        """
        Return forecast DataFrames for each model.

        Each DataFrame has a ``date`` column followed by triplets of columns
        per target: ``{col}_yhat``, ``{col}_yhat_lower``, ``{col}_yhat_upper``.

        :return: Dictionary with keys ``"prophet"`` and ``"arima"``, each
                 mapping to a forecast DataFrame of length ``date_pred``.
        :raises RuntimeError: If ``fit()`` has not been called yet.
        """
        if self._forecast is None:
            raise RuntimeError("No forecast available. Call fit() first.")
        return self._forecast