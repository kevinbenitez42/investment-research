"""Configuration-driven loaders for country macroeconomic notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import pandas as pd

from Quantapp.analytics.macro import transform_macro_series
from Quantapp.data.macro_data_client import MacroDataClient


@dataclass(frozen=True)
class IndicatorSpec:
    key: str
    label: str
    series_id: str
    section: str
    unit: str
    view: str = "level"
    periods: int | None = None
    max_age_days: int | None = None
    zero_line: bool = False
    source_note: str = "FRED"


@dataclass(frozen=True)
class CountryMacroConfig:
    slug: str
    name: str
    currency: str
    indicators: tuple[IndicatorSpec, ...]
    cycle_series_id: str | None = None
    default_years: int = 20
    source_caveat: str | None = None


@dataclass
class CountryMacroData:
    config: CountryMacroConfig
    raw: pd.DataFrame
    series: dict[str, pd.Series]
    cycle: pd.Series | None
    summary: pd.DataFrame
    warnings: list[str]
    fetch_errors: dict[str, str]


def _i(key, label, series_id, section, unit, view="level", periods=None,
       max_age_days=None, zero_line=False, source_note="FRED"):
    return IndicatorSpec(
        key, label, series_id, section, unit, view, periods, max_age_days,
        zero_line, source_note,
    )


GROWTH = "Growth and Business Activity"
LABOR = "Labor and Domestic Demand"
INFLATION = "Inflation and Monetary Policy"
EXTERNAL = "Rates, Currency and External Balance"
FOCUS = "Country-Specific Risk Indicators"


COUNTRY_MACRO_CONFIGS = {
    "australia": CountryMacroConfig(
        "australia", "Australia", "AUD", (
            _i("real_gdp", "Real GDP - YoY", "NGDPRSAXDCAUQ", GROWTH, "%", "yoy", 4, 240),
            _i("industrial_production", "Industrial Production - YoY", "AUSPROINDQISMEI", GROWTH, "%", "yoy", 4, 240),
            _i("unemployment", "Unemployment Rate", "LRUNTTTTAUM156S", LABOR, "%", max_age_days=120),
            _i("cpi", "Consumer Prices - YoY", "AUSCPIALLQINMEI", INFLATION, "%", "yoy", 4, 240),
            _i("policy_rate", "Interbank/Call Rate Proxy", "IRSTCI01AUM156N", INFLATION, "%", max_age_days=120),
            _i("long_yield", "10-Year Government Yield", "IRLTLT01AUM156N", EXTERNAL, "%", max_age_days=120),
            _i("fx_strength", "AUD Strength vs USD - 12M", "DEXUSAL", EXTERNAL, "%", "currency_strength", 252, 14, True),
            _i("current_account", "Current Account", "AUSBCAGDPBP6PT", EXTERNAL, "% GDP", max_age_days=550),
            _i("reer", "Real Effective Exchange Rate - YoY", "RBAUBIS", FOCUS, "%", "yoy", 12, 120, True),
            _i("household_debt", "Household Debt to GDP", "HDTGPDAUQ163N", FOCUS, "% GDP", max_age_days=240),
            _i("real_house_prices", "Real House Prices - YoY", "QAUR628BIS", FOCUS, "%", "yoy", 4, 240),
        ), "AUSRECDM",
        source_caveat="Industrial production and some OECD series may lag; use ABS/RBA as the current-data fallback.",
    ),
    "brazil": CountryMacroConfig(
        "brazil", "Brazil", "BRL", (
            _i("real_gdp", "Real GDP - YoY", "NGDPRSAXDCBRQ", GROWTH, "%", "yoy", 4, 240),
            _i("industrial_production", "Industrial Production - YoY", "BRAPROINDMISMEI", GROWTH, "%", "yoy", 12, 120),
            _i("cpi", "Consumer Prices - YoY", "BRACPIALLMINMEI", INFLATION, "%", "yoy", 12, 120),
            _i("policy_rate", "Discount Rate Proxy", "INTDSRBRM193N", INFLATION, "%", max_age_days=120),
            _i("fx_strength", "BRL Strength vs USD - 12M", "DEXBZUS", EXTERNAL, "%", "inverse_currency_strength", 252, 14, True),
            _i("current_account", "Current Account", "BRABCAGDPBP6", EXTERNAL, "% GDP", max_age_days=550),
            _i("reer", "Real Effective Exchange Rate - YoY", "RBBRBIS", FOCUS, "%", "yoy", 12, 120, True),
            _i("reserves", "Foreign Exchange Reserves - YoY", "TRESEGBRM052N", FOCUS, "%", "yoy", 12, 120, True),
        ), "BRARECDM",
        source_caveat="FRED labor and some activity series are stale; use IBGE and BCB for live Brazilian releases.",
    ),
    "canada": CountryMacroConfig(
        "canada", "Canada", "CAD", (
            _i("real_gdp", "Real GDP - YoY", "NGDPRSAXDCCAQ", GROWTH, "%", "yoy", 4, 240),
            _i("industrial_production", "Industrial Production - YoY", "CANPROINDMISMEI", GROWTH, "%", "yoy", 12, 120),
            _i("unemployment", "Unemployment Rate", "LRUNTTTTCAM156S", LABOR, "%", max_age_days=120),
            _i("cpi", "Consumer Prices - YoY", "CANCPIALLMINMEI", INFLATION, "%", "yoy", 12, 120),
            _i("policy_rate", "Overnight Rate Proxy", "IRSTCI01CAM156N", INFLATION, "%", max_age_days=120),
            _i("long_yield", "10-Year Government Yield", "IRLTLT01CAM156N", EXTERNAL, "%", max_age_days=120),
            _i("fx_strength", "CAD Strength vs USD - 12M", "DEXCAUS", EXTERNAL, "%", "inverse_currency_strength", 252, 14, True),
            _i("current_account", "Current Account", "CANBCAGDPBP6", EXTERNAL, "% GDP", max_age_days=550),
            _i("reer", "Real Effective Exchange Rate - YoY", "RBCABIS", FOCUS, "%", "yoy", 12, 120, True),
            _i("household_debt", "Household Debt to GDP", "HDTGPDCAQ163N", FOCUS, "% GDP", max_age_days=240),
            _i("real_house_prices", "Real House Prices - YoY", "QCAR628BIS", FOCUS, "%", "yoy", 4, 240),
        ), "CANRECDM",
        source_caveat="Use Statistics Canada and the Bank of Canada when an OECD/FRED series fails freshness checks.",
    ),
    "china": CountryMacroConfig(
        "china", "China", "CNY", (
            _i("real_gdp", "Real GDP - YoY", "NGDPRXDCCNA", GROWTH, "%", "yoy", 1, 550),
            _i("industrial_production", "Industrial Production - YoY", "PRINTO01CNQ663N", GROWTH, "%", "yoy", 4, 240),
            _i("cpi", "Consumer Prices - YoY", "CHNCPIALLMINMEI", INFLATION, "%", "yoy", 12, 120),
            _i("policy_rate", "3-Month Interbank Rate", "IR3TIB01CNM156N", INFLATION, "%", max_age_days=120),
            _i("fx_strength", "CNY Strength vs USD - 12M", "DEXCHUS", EXTERNAL, "%", "inverse_currency_strength", 252, 14, True),
            _i("current_account", "Current Account", "CHNB6BLTT02STSAQ", EXTERNAL, "% GDP", max_age_days=240),
            _i("reer", "Real Effective Exchange Rate - YoY", "RBCNBIS", FOCUS, "%", "yoy", 12, 120, True),
            _i("reserves", "Foreign Exchange Reserves - YoY", "TRESEGCNM052N", FOCUS, "%", "yoy", 12, 120, True),
        ), "CHNRECDM",
        source_caveat="Use NBS and PBoC for current property, credit, CPI and activity releases; several FRED feeds lag.",
    ),
    "euro-area": CountryMacroConfig(
        "euro-area", "Euro Area", "EUR", (
            _i("real_gdp", "Real GDP - YoY", "CLVMEURSCAB1GQEA19", GROWTH, "%", "yoy", 4, 240),
            _i("industrial_production", "Industrial Production - YoY", "EA19PRINTO01IXOBSAM", GROWTH, "%", "yoy", 12, 120),
            _i("unemployment", "Unemployment Rate", "LRHUTTTTEZM156S", LABOR, "%", max_age_days=120),
            _i("cpi", "HICP - YoY", "CP0000EZ19M086NEST", INFLATION, "%", "yoy", 12, 120),
            _i("policy_rate", "ECB Deposit Facility Rate", "ECBDFR", INFLATION, "%", max_age_days=14),
            _i("long_yield", "Euro-Area 10-Year Yield", "IRLTLT01EZM156N", EXTERNAL, "%", max_age_days=120),
            _i("fx_strength", "EUR Strength vs USD - 12M", "DEXUSEU", EXTERNAL, "%", "currency_strength", 252, 14, True),
            _i("current_account", "Current Account", "EA19B6BLTT02STSAQ", EXTERNAL, "% GDP", max_age_days=240),
            _i("ecb_assets", "ECB Total Assets", "ECBASSETSW", FOCUS, "EUR millions", max_age_days=30),
            _i("high_yield_spread", "Euro High-Yield OAS", "BAMLHE00EHYIOAS", FOCUS, "%", max_age_days=14),
        ), "EURORECDM",
        source_caveat="Bloc aggregates hide core/periphery divergence; add national spreads for allocation decisions.",
    ),
    "india": CountryMacroConfig(
        "india", "India", "INR", (
            _i("real_gdp", "Real GDP - YoY", "NGDPRNSAXDCINQ", GROWTH, "%", "yoy", 4, 240),
            _i("industrial_production", "Manufacturing Production - YoY", "INDPRMNTO01GYSAM", GROWTH, "%", max_age_days=120),
            _i("cpi", "Consumer Prices - YoY", "INDCPIALLMINMEI", INFLATION, "%", "yoy", 12, 120),
            _i("policy_rate", "3-Month Interbank Rate", "INDIR3TIB01STM", INFLATION, "%", max_age_days=120),
            _i("long_yield", "10-Year Government Yield", "INDIRLTLT01STM", EXTERNAL, "%", max_age_days=120),
            _i("fx_strength", "INR Strength vs USD - 12M", "DEXINUS", EXTERNAL, "%", "inverse_currency_strength", 252, 14, True),
            _i("current_account", "Current Account", "INDBCAGDPBP6PT", EXTERNAL, "% GDP", max_age_days=550),
            _i("reer", "Real Effective Exchange Rate - YoY", "RBINBIS", FOCUS, "%", "yoy", 12, 120, True),
            _i("reserves", "Foreign Exchange Reserves - YoY", "TRESEGINM052N", FOCUS, "%", "yoy", 12, 120, True),
        ), "INDRECDM",
        source_caveat="Use MOSPI and RBI for current food inflation, policy, liquidity and bank-credit releases.",
    ),
    "japan": CountryMacroConfig(
        "japan", "Japan", "JPY", (
            _i("real_gdp", "Real GDP - YoY", "JPNRGDPEXP", GROWTH, "%", "yoy", 4, 240),
            _i("industrial_production", "Industrial Production - YoY", "JPNPROINDMISMEI", GROWTH, "%", "yoy", 12, 120),
            _i("unemployment", "Unemployment Rate", "LRUN64TTJPM156S", LABOR, "%", max_age_days=120),
            _i("cpi", "Consumer Prices - YoY", "JPNCPIALLMINMEI", INFLATION, "%", "yoy", 12, 120),
            _i("policy_rate", "Call Rate Proxy", "IRSTCI01JPM156N", INFLATION, "%", max_age_days=120),
            _i("long_yield", "10-Year Government Yield", "IRLTLT01JPM156N", EXTERNAL, "%", max_age_days=120),
            _i("fx_strength", "JPY Strength vs USD - 12M", "DEXJPUS", EXTERNAL, "%", "inverse_currency_strength", 252, 14, True),
            _i("current_account", "Current Account", "JPNB6BLTT02STSAQ", EXTERNAL, "% GDP", max_age_days=240),
            _i("boj_assets", "Bank of Japan Assets", "JPNASSETS", FOCUS, "JPY 100 millions", max_age_days=30),
            _i("reer", "Real Effective Exchange Rate - YoY", "RBJPBIS", FOCUS, "%", "yoy", 12, 120, True),
            _i("real_house_prices", "Real House Prices - YoY", "QJPR628BIS", FOCUS, "%", "yoy", 4, 240),
        ), "JPNRECDM",
        source_caveat="FRED's Japan CPI feed is materially stale; use Statistics Bureau/BOJ before current-policy conclusions.",
    ),
    "mexico": CountryMacroConfig(
        "mexico", "Mexico", "MXN", (
            _i("real_gdp", "Real GDP - YoY", "NGDPRSAXDCMXQ", GROWTH, "%", "yoy", 4, 240),
            _i("industrial_production", "Manufacturing Production - YoY", "MEXPRMNTO01GYSAM", GROWTH, "%", max_age_days=120),
            _i("unemployment", "Unemployment Rate", "LRUNTTTTMXQ156S", LABOR, "%", max_age_days=240),
            _i("cpi", "Consumer Prices - YoY", "MEXCPIALLMINMEI", INFLATION, "%", "yoy", 12, 120),
            _i("policy_rate", "3-Month Interbank Rate", "IR3TIB01MXM156N", INFLATION, "%", max_age_days=120),
            _i("long_yield", "10-Year Government Yield", "IRLTLT01MXM156N", EXTERNAL, "%", max_age_days=120),
            _i("fx_strength", "MXN Strength vs USD - 12M", "DEXMXUS", EXTERNAL, "%", "inverse_currency_strength", 252, 14, True),
            _i("current_account", "Current Account", "MEXB6BLTT02STSAQ", EXTERNAL, "% GDP", max_age_days=240),
            _i("reer", "Real Effective Exchange Rate - YoY", "RBMXBIS", FOCUS, "%", "yoy", 12, 120, True),
            _i("reserves", "Foreign Exchange Reserves - YoY", "TRESEGMXM052N", FOCUS, "%", "yoy", 12, 120, True),
        ), "MEXRECDM",
        source_caveat="Use INEGI and Banxico for current inflation, policy, remittance, trade and nearshoring data.",
    ),
    "saudi-arabia": CountryMacroConfig(
        "saudi-arabia", "Saudi Arabia", "SAR", (
            _i("real_gdp", "Real GDP - YoY", "NGDPRSAXDCSAQ", GROWTH, "%", "yoy", 4, 240),
            _i("non_oil_growth", "Non-Oil Real GDP Growth", "SAUNGDPXORPCHPT", GROWTH, "%", max_age_days=550),
            _i("cpi", "Consumer Prices - YoY", "SAUCPALTT01IXOBM", INFLATION, "%", "yoy", 12, 120),
            _i("fx_strength", "SAR Strength vs USD - 12M", "SAUCCUSMA02STM", EXTERNAL, "%", "inverse_currency_strength", 12, 120, True),
            _i("current_account", "Current Account", "SAUBCAGDPGDPPT", EXTERNAL, "% GDP", max_age_days=550),
            _i("reer", "Real Effective Exchange Rate - YoY", "RBSABIS", FOCUS, "%", "yoy", 12, 120, True),
            _i("reserves", "Foreign Exchange Reserves - YoY", "TRESEGSAM052N", FOCUS, "%", "yoy", 12, 120, True),
            _i("oil_production", "Oil Production", "SAUNGDPMOMBD", FOCUS, "million bbl/day", max_age_days=550),
            _i("fiscal_breakeven", "Fiscal Breakeven Oil Price", "SAUPZPIOILBEGUSD", FOCUS, "USD/bbl", max_age_days=550),
        ), None,
        source_caveat="Use GASTAT and SAMA for current releases; the SAR peg makes reserves and oil/fiscal flows more informative than spot FX.",
    ),
    "south-korea": CountryMacroConfig(
        "south-korea", "South Korea", "KRW", (
            _i("real_gdp", "Real GDP - YoY", "NGDPRSAXDCKRQ", GROWTH, "%", "yoy", 4, 240),
            _i("industrial_production", "Industrial Production - YoY", "KORPROINDMISMEI", GROWTH, "%", "yoy", 12, 120),
            _i("unemployment", "Unemployment Rate", "LRUNTTTTKRM156S", LABOR, "%", max_age_days=120),
            _i("cpi", "Consumer Prices - YoY", "KORCPIALLMINMEI", INFLATION, "%", "yoy", 12, 120),
            _i("policy_rate", "Call Rate Proxy", "IRSTCI01KRM156N", INFLATION, "%", max_age_days=120),
            _i("long_yield", "10-Year Government Yield", "IRLTLT01KRM156N", EXTERNAL, "%", max_age_days=120),
            _i("fx_strength", "KRW Strength vs USD - 12M", "DEXKOUS", EXTERNAL, "%", "inverse_currency_strength", 252, 14, True),
            _i("current_account", "Current Account", "KORB6BLTT02STSAQ", EXTERNAL, "% GDP", max_age_days=240),
            _i("reer", "Real Effective Exchange Rate - YoY", "RBKRBIS", FOCUS, "%", "yoy", 12, 120, True),
            _i("real_house_prices", "Real House Prices - YoY", "QKRR628BIS", FOCUS, "%", "yoy", 4, 240),
        ), "KORRECDM",
        source_caveat="Use Bank of Korea and Korea Customs for current semiconductor, export, policy and CPI data.",
    ),
    "united-kingdom": CountryMacroConfig(
        "united-kingdom", "United Kingdom", "GBP", (
            _i("real_gdp", "Real GDP - YoY", "NGDPRSAXDCGBQ", GROWTH, "%", "yoy", 4, 240),
            _i("industrial_production", "Industrial Production - YoY", "GBRPROINDMISMEI", GROWTH, "%", "yoy", 12, 120),
            _i("unemployment", "Unemployment Rate", "LRHUTTTTGBM156S", LABOR, "%", max_age_days=120),
            _i("cpi", "Consumer Prices - YoY", "GBRCPIALLMINMEI", INFLATION, "%", "yoy", 12, 120),
            _i("policy_rate", "Interbank Rate Proxy", "IRSTCI01GBM156N", INFLATION, "%", max_age_days=120),
            _i("long_yield", "10-Year Gilt Yield", "IRLTLT01GBM156N", EXTERNAL, "%", max_age_days=120),
            _i("fx_strength", "GBP Strength vs USD - 12M", "DEXUSUK", EXTERNAL, "%", "currency_strength", 252, 14, True),
            _i("current_account", "Current Account", "GBRB6BLTT02STSAQ", EXTERNAL, "% GDP", max_age_days=240),
            _i("reer", "Real Effective Exchange Rate - YoY", "RBGBBIS", FOCUS, "%", "yoy", 12, 120, True),
            _i("household_debt", "Household Debt to GDP", "HDTGPDGBQ163N", FOCUS, "% GDP", max_age_days=240),
            _i("real_house_prices", "Real House Prices - YoY", "QGBR628BIS", FOCUS, "%", "yoy", 4, 240),
        ), "GBRRECDM",
        source_caveat="The configured interbank rate is a policy proxy; use the Bank of England API for the live Bank Rate.",
    ),
}


def get_country_macro_config(slug: str) -> CountryMacroConfig:
    normalized = slug.strip().lower().replace("_", "-").replace(" ", "-")
    try:
        return COUNTRY_MACRO_CONFIGS[normalized]
    except KeyError as exc:
        available = ", ".join(COUNTRY_MACRO_CONFIGS)
        raise KeyError(f"Unknown country macro config {slug!r}. Available: {available}") from exc


def list_country_macro_configs() -> tuple[str, ...]:
    return tuple(COUNTRY_MACRO_CONFIGS)


def _freshness_warning(indicator: IndicatorSpec, series: pd.Series) -> str | None:
    if indicator.max_age_days is None or series.dropna().empty:
        return None
    latest = pd.Timestamp(series.dropna().index.max()).tz_localize(None)
    age_days = (pd.Timestamp(date.today()) - latest.normalize()).days
    if age_days <= indicator.max_age_days:
        return None
    return (
        f"{indicator.label} ({indicator.series_id}) is stale: latest observation "
        f"is {latest.date()} ({age_days} days old)."
    )


def load_country_macro(
    config: CountryMacroConfig | str,
    *,
    fred_key: str | None = None,
    start_date: str | None = "1960-01-01",
) -> CountryMacroData:
    """Fetch, transform and freshness-check one configured economy."""
    if isinstance(config, str):
        config = get_country_macro_config(config)
    client = MacroDataClient(fred_key=fred_key)

    unique_ids = dict.fromkeys(indicator.series_id for indicator in config.indicators)
    if config.cycle_series_id:
        unique_ids[config.cycle_series_id] = None
    raw = client.fetch_fred_series(
        {series_id: series_id for series_id in unique_ids},
        start_date=start_date,
        on_error="ignore",
    )
    fetch_errors = dict(raw.attrs.get("fetch_errors", {}))

    transformed = {}
    warnings = []
    summary_rows = []
    for indicator in config.indicators:
        if indicator.series_id not in raw:
            continue
        values = transform_macro_series(
            raw[indicator.series_id],
            indicator.view,
            periods=indicator.periods,
        )
        transformed[indicator.key] = values.rename(indicator.label)
        stale = _freshness_warning(indicator, raw[indicator.series_id])
        if stale:
            warnings.append(stale)
        if not values.empty:
            summary_rows.append({
                "Section": indicator.section,
                "Indicator": indicator.label,
                "Latest": values.iloc[-1],
                "As of": values.index[-1],
                "Unit": indicator.unit,
                "Source": indicator.source_note,
                "Freshness": "STALE" if stale else "Current",
            })

    cycle = None
    if config.cycle_series_id and config.cycle_series_id in raw:
        cycle = raw[config.cycle_series_id].dropna().rename("Cycle contraction")
        warnings.append(
            "Cycle shading uses a discontinued OECD historical indicator; it is "
            "contextual and is not a live recession signal."
        )
    if config.source_caveat:
        warnings.append(config.source_caveat)
    for series_id, error in fetch_errors.items():
        warnings.append(f"Could not load {series_id}: {error}")

    return CountryMacroData(
        config=config,
        raw=raw,
        series=transformed,
        cycle=cycle,
        summary=pd.DataFrame(summary_rows),
        warnings=warnings,
        fetch_errors=fetch_errors,
    )


__all__ = [
    "COUNTRY_MACRO_CONFIGS",
    "CountryMacroConfig",
    "CountryMacroData",
    "IndicatorSpec",
    "get_country_macro_config",
    "list_country_macro_configs",
    "load_country_macro",
]
