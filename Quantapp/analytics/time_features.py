"""Calendar and seasonality feature utilities."""

from __future__ import annotations

import holidays
import pandas as pd
from statsmodels.tsa.seasonal import STL


class TimeFeatures:
    """Time-domain feature generation helpers."""

    def holiday_features(self, df, country="US"):
        """Alias for holiday feature generation without the legacy `compute_` prefix."""
        return self.compute_holiday_features(df, country=country)

    def compute_holiday_features(self, df, country="US"):
        """
        Add holiday flags/ids and countdown features for each holiday type.

        Returns a DataFrame indexed like `df`.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame or Series must have a DatetimeIndex")

        start_year = df.index.year.min()
        end_year = df.index.year.max()

        holiday_names = {}
        holiday_id_map = {}
        holiday_counter = 1
        holiday_dates_map = {}

        for year in range(start_year, end_year + 1):
            country_holidays = holidays.CountryHoliday(country, years=[year])
            for date, name in country_holidays.items():
                timestamp_date = pd.Timestamp(date)
                holiday_names[timestamp_date] = name
                if name not in holiday_id_map:
                    holiday_id_map[name] = holiday_counter
                    holiday_counter += 1
                holiday_dates_map.setdefault(name, []).append(timestamp_date)

        for name in holiday_dates_map:
            holiday_dates_map[name] = sorted(holiday_dates_map[name])

        is_holiday = df.index.map(lambda x: 1 if x in holiday_names else 0)
        holiday_id = df.index.map(lambda x: holiday_id_map.get(holiday_names.get(x, ""), 0))

        countdown_columns = {f"days_until_{name}": [] for name in holiday_dates_map}
        for date in df.index:
            for name, dates in holiday_dates_map.items():
                days_until = None
                for holiday_date in dates:
                    if holiday_date >= date:
                        days_until = (holiday_date - date).days
                        break
                if days_until is None:
                    next_year_start = pd.Timestamp(year=date.year + 1, month=1, day=1)
                    days_until = (next_year_start - date).days
                countdown_columns[f"days_until_{name}"].append(days_until)

        holiday_dates_sorted = sorted(holiday_names.keys())
        days_until_next_holiday = []
        for x in df.index:
            future_dates = [(holiday_date - x).days for holiday_date in holiday_dates_sorted if holiday_date > x]
            if future_dates:
                days_until_next_holiday.append(min(future_dates))
            else:
                next_year_start = pd.Timestamp(year=x.year + 1, month=1, day=1)
                days_until_next_holiday.append((next_year_start - x).days)

        holidays_df = pd.DataFrame(
            {
                "date": df.index,
                "is_holiday": is_holiday,
                "holiday_id": holiday_id,
                "days_until_next_holiday": days_until_next_holiday,
            }
        ).set_index("date")

        for name, countdown in countdown_columns.items():
            holidays_df[f"days_until_{name}"] = countdown

        return holidays_df

    def seasonal_decompositions(self, df, seasonal_periods=(5, 21, 63, 125, 253)):
        """Alias for STL decomposition without the legacy `compute_` prefix."""
        return self.compute_seasonal_decompositions(df, seasonal_periods=seasonal_periods)

    def compute_seasonal_decompositions(self, df, seasonal_periods=(5, 21, 63, 125, 253)):
        """
        Compute STL trend/seasonal/residual decomposition per column and period.
        """
        decompositions = pd.DataFrame(index=df.index)

        for period in seasonal_periods:
            for col in df.columns:
                series = df[col]
                stl = STL(series, seasonal=period)
                result = stl.fit()
                decompositions[f"{col}_trend_{period}"] = result.trend
                decompositions[f"{col}_seasonal_{period}"] = result.seasonal
                decompositions[f"{col}_residual_{period}"] = result.resid

        return decompositions
