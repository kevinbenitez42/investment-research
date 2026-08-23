import pandas as pd


class Helper:
    def simplify_datetime_index(self,series):
        """
        Simplifies the DateTime index of a Series to contain only the date (YYYY-MM-DD),
        maintaining it as a DateTimeIndex without timezone information.
        
        Parameters:
            series (pd.Series): The input Series with a DateTimeIndex.
        
        Returns:
            pd.Series: The Series with the DateTime index simplified to YYYY-MM-DD.
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("The Series index must be a DateTimeIndex.")
        
        # Remove timezone information if present
        if series.index.tz is not None:
            series = series.copy()
            series.index = series.index.tz_convert('UTC').tz_localize(None)
        
        # Normalize the index to remove the time component
        series.index = series.index.normalize()
        
        return series

    @staticmethod
    def is_futures_ticker(ticker_str):
        symbol = str(ticker_str).strip().upper()
        return symbol.endswith("=F")

    def build_equity_like_trade_range_source(
        self,
        ticker_str,
        daily_frame,
        *,
        intraday_frame=None,
        session_timezone="America/New_York",
    ):
        """
        Build the Open/Close source used by trade-range style analytics.

        Equities keep the supplied daily bars. Futures keep their full daily history too,
        but callers may attach NY cash-session intraday data for the latest session so
        the current cone can anchor off 09:30-16:00 ET data without collapsing the history
        panel to the recent intraday download window.
        """
        if not isinstance(daily_frame, pd.DataFrame):
            raise TypeError("daily_frame must be a pandas DataFrame.")
        if "Open" not in daily_frame.columns or "Close" not in daily_frame.columns:
            raise ValueError("daily_frame must contain 'Open' and 'Close' columns.")

        base_frame = daily_frame.copy()
        base_frame.attrs["source_ticker"] = str(ticker_str).upper()
        if not self.is_futures_ticker(ticker_str):
            base_frame.attrs["session_mode"] = "daily_bar"
            return base_frame

        base_frame.attrs["session_mode"] = "futures_daily_bar"

        if intraday_frame is None:
            base_frame.attrs["session_mode"] = "futures_daily_bar_intraday_fallback"
            return base_frame
        if not isinstance(intraday_frame, pd.DataFrame):
            raise TypeError("intraday_frame must be a pandas DataFrame when provided.")
        if intraday_frame.empty:
            base_frame.attrs["session_mode"] = "futures_daily_bar_intraday_fallback"
            return base_frame

        intraday_frame = intraday_frame.copy()
        if not isinstance(intraday_frame.index, pd.DatetimeIndex):
            intraday_frame.index = pd.to_datetime(intraday_frame.index)

        if intraday_frame.index.tz is None:
            intraday_frame.index = intraday_frame.index.tz_localize(session_timezone)
        else:
            intraday_frame.index = intraday_frame.index.tz_convert(session_timezone)

        intraday_frame = intraday_frame.sort_index()
        intraday_frame = intraday_frame[intraday_frame.index.dayofweek < 5]
        intraday_frame = intraday_frame.between_time("09:30", "16:00", inclusive="both")

        required_columns = [column for column in ("Open", "High", "Low", "Close") if column in intraday_frame.columns]
        if "Open" not in required_columns or "Close" not in required_columns:
            base_frame.attrs["session_mode"] = "futures_daily_bar_intraday_missing_columns"
            return base_frame

        if intraday_frame.empty:
            base_frame.attrs["session_mode"] = "futures_daily_bar_intraday_empty_session"
            return base_frame

        aggregation_map = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in intraday_frame.columns:
            aggregation_map["Volume"] = "sum"

        session_frame = (
            intraday_frame.groupby(intraday_frame.index.normalize())
            .agg(aggregation_map)
            .dropna(subset=["Open", "Close"])
            .sort_index()
        )
        if session_frame.empty:
            base_frame.attrs["session_mode"] = "futures_daily_bar_intraday_no_sessions"
            return base_frame

        session_frame.index = pd.DatetimeIndex(session_frame.index).tz_localize(None)
        latest_session = session_frame.iloc[-1]

        base_frame.attrs["session_mode"] = "hybrid_daily_history_ny_anchor"
        base_frame.attrs["current_session_anchor_price"] = float(latest_session["Open"])
        base_frame.attrs["current_session_reference_price"] = float(base_frame["Close"].iloc[-1])
        base_frame.attrs["current_session_latest_price"] = float(latest_session["Close"])
        base_frame.attrs["current_session_date"] = pd.Timestamp(session_frame.index[-1])
        base_frame.attrs["trade_range_recent_session_count"] = int(len(session_frame))
        return base_frame
