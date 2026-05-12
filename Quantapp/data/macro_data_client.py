import datetime
import os

try:
    import nasdaqdatalink
except ImportError:
    nasdaqdatalink = None

import pandas as pd

from Quantapp.data.sources.fred import (
    fetch_fred_observations_query,
    fetch_historical_treasury_yields,
    fetch_fred_series_frame,
    format_fred_date,
    fred_api_url,
    resolve_fred_api_key,
)
from Quantapp.data.sources.yfinance_history import fetch_history

class MacroDataClient:
    def __init__(self, fred_key=None):
        self.start_date = datetime.datetime(1900, 1, 1)
        self.end_date = datetime.datetime.now()  # Current date
        self.fred_api_key = fred_key or resolve_fred_api_key()
        self._configure_nasdaq_data_link()

    def _configure_nasdaq_data_link(self):
        if nasdaqdatalink is None:
            return

        api_key = (
            os.getenv("NASDAQ_DATA_LINK_API_KEY")
            or os.getenv("NASDAQ_DATALINK_API_KEY")
            or os.getenv("QUANDL_API_KEY")
        )
        proxy = os.getenv("NASDAQ_DATA_LINK_PROXY") or os.getenv("NASDAQ_DATALINK_PROXY")

        if api_key:
            nasdaqdatalink.ApiConfig.api_key = api_key
        if proxy:
            nasdaqdatalink.ApiConfig.proxy = proxy

    def _require_nasdaq_data_link(self):
        if nasdaqdatalink is None:
            raise ImportError("nasdaqdatalink is required for this method.")
        return nasdaqdatalink

    def _format_fred_date(self, value):
        if value is None:
            return None
        return format_fred_date(value)

    def base_url(self, series_id, start_date=None, end_date=None):
        " Constructs the base URL for FRED API calls."
        if not self.fred_api_key:
            raise ValueError("A FRED API key is required. Pass fred_key or set FRED_API_KEY.")
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
        }
        if start_date is not None:
            params["observation_start"] = self._format_fred_date(start_date)
        if end_date is not None:
            params["observation_end"] = self._format_fred_date(end_date)
        return fred_api_url("series/observations", params=params)
    
    def fetch_fred_json(self,query):
        """
        Fetches JSON data from the provided FRED API query URL.
        
        Args:
            query (str): The FRED API query URL.
        
        Returns:
            pd.DataFrame: Parsed FRED observations with a datetime index.
            
        Raises:
            HTTPError: If the HTTP request returned an unsuccessful status code.
            RequestException: For other request-related errors.
        """
        return fetch_fred_observations_query(query)
        
    def get_inflation_data(self):
        series_ids = {
            "CPI": "CPIAUCSL",
            "PPI": "PPIACO",
            "Core CPI": "CPILFESL",
            "Core PPI": "WPSFD4131",
            "Core PCE": "PCEPILFE",
            "CPI Commodities": "CPIAPPSL",
            "CPI Energy": "CPIENGSL",
            "CPI Food": "CPIFABSL",
            #"CPI Services": "CPISERVSL",
            "CPI Shelter": "CPIHOSSL",
            "CPI Rent of Primary Residence": "CUSR0000SEHA",
            "CPI Owners' Equivalent Rent": "CUSR0000SEHC"
        }
    
        return fetch_fred_series_frame(series_ids, api_key=self.fred_api_key)
 
    def get_interest_rate_data(self):
        series_ids = {
            "Federal Funds Rate": "FEDFUNDS",
            "10-Year Treasury Constant Maturity Rate": "DGS10",
            "10-Year TIPS Rate": "DFII10",
            "AAA Corporate Bond Yield": "DAAA",
            "BAA Corporate Bond Yield": "DBAA"
        }
        
        return fetch_fred_series_frame(series_ids, api_key=self.fred_api_key)

    def get_historical_treasury_yields(self, maturities=None, start_date=None, end_date=None, real=False):
        """
        Fetch historical US Treasury constant maturity yields from FRED.

        Args:
            maturities (str | list[str] | tuple[str] | None): Treasury maturities to fetch.
                Uses all available maturities when omitted.
            start_date (str | datetime.date | datetime.datetime | None): Optional start date.
            end_date (str | datetime.date | datetime.datetime | None): Optional end date.
            real (bool): When True, fetch real Treasury yields instead of nominal yields.

        Returns:
            pd.DataFrame: Treasury yield history indexed by date.
        """
        return fetch_historical_treasury_yields(
            maturities=maturities,
            api_key=self.fred_api_key,
            start_date=start_date,
            end_date=end_date,
            real=real,
        )
    
    def get_gdp_data(self):
        series_ids = {
            # GDP high level
            "Nominal GDP": "GDP",
            "Real GDP": "GDPC1",
             
            #nominal GDP components
            "Consumption Expenditures": "PCE",
            "Investment": "GPDI",
            "Government Spending": "GCE",
            "Exports": "EXPGS",
            "Imports": "IMPGS",
            "Net Exports": "NETEXP",
            
            # real GDP components
            "Real Consumption Expenditures": "PCEC96",
            "Real Investment": "GPDIC1",
            "Real Government Spending": "GCEC1",
            "Real Exports": "EXPGSC1",
            "Real Imports": "IMPGSC1",
            "Real Net Exports": "NETEXC"
        }
        
        return fetch_fred_series_frame(series_ids, api_key=self.fred_api_key)

    def get_recession_indicators(self):
        series_ids = {
            "NBER Recession Indicators": "USREC",
            "OECD Recession Indicators": "USARECDM",
            "Real-Time Sahm Rule": "SAHMREALTIME",
            "Markov Switching Smoothed Probability": "RECPROUSM156N"
        }
        return fetch_fred_series_frame(series_ids, api_key=self.fred_api_key)
    
    def get_bond_data(self):
        ndl = self._require_nasdaq_data_link()
        return {
            'treasury yield curve rates': ndl.get("USTREASURY/YIELD"),
            'treasury yield curve rates (real)': ndl.get("USTREASURY/REALYIELD"),
            'investment grade corporate bond yield curve rates': ndl.get("USTREASURY/HQMYC"),
            'US High yield Option-Adjusted Spread': ndl.get("FRED/BAMLH0A0HYM2"),
            'Treasury' : fetch_history('GOVT', period='max', interval='1d'),
            'Investment Corporate Bonds' : fetch_history('LQD', period='max', interval='1d'),
            'High yield Corporate Bonds' : fetch_history('HYG', period='max', interval='1d')
        }
    
    def get_housing_market_data(self):
        return {
            'Building Permits': fetch_fred_series_frame(
                {"PERMIT": "PERMIT"},
                api_key=self.fred_api_key,
                start_date=self.start_date,
                end_date=self.end_date,
            ),
            'Housing Starts': fetch_fred_series_frame(
                {"HOUST": "HOUST"},
                api_key=self.fred_api_key,
                start_date=self.start_date,
                end_date=self.end_date,
            ),
            'New Home Sales': fetch_fred_series_frame(
                {"HSN1F": "HSN1F"},
                api_key=self.fred_api_key,
                start_date=self.start_date,
                end_date=self.end_date,
            )
           # 'Existing Home Sales' : nasdaqdatalink.get("FRED/EXHOSLUSM495S"),
           # 'Case Shiller Home Price Index' : nasdaqdatalink.get("FRED/CSUSHPISA")
        }

    def get_leading_indicators(self):
        ndl = self._require_nasdaq_data_link()
        ism_new_order_index                   = ndl.get("ISM/MAN_NEWORDERS")
        average_weekly_hours_manufacturing    = ndl.get("FRED/PRS84006023")
        initial_claims                        = ndl.get("FRED/ICSA")
        manufacturers_new_orders_ex_aircraft  = ndl.get("FRED/AMXDNO")
        manufacturers_new_orders_consumer_goods= ndl.get("FRED/ACOGNO")
        leading_credit_index                   = ndl.get("FRED/USSLIND")
        ten_year_minus_federal_funds_rate_monthly     = ndl.get("FRED/T10YFFM")
     #   consumer_sentiment                            = nasdaqdatalink.get("UMICH/SOC1")
        return {
            "ism_new_order_index"                   : ism_new_order_index,
            "average_weekly_hours_manufacturing"    : pd.Series(average_weekly_hours_manufacturing['Value'],index=average_weekly_hours_manufacturing.index),
            "initial_claims"                        : pd.Series(initial_claims['Value'],index=initial_claims.index),
            "manufacturers_new_orders_ex_aircraft"  : pd.Series(manufacturers_new_orders_ex_aircraft['Value'],index=manufacturers_new_orders_ex_aircraft.index),
            "manufacturers_new_orders_consumer_goods": pd.Series(manufacturers_new_orders_consumer_goods['Value'],index=manufacturers_new_orders_consumer_goods.index),
            "leading_credit_index"                   : pd.Series(leading_credit_index['Value'],index=leading_credit_index.index),
            "ten_year_minus_federal_funds_rate_monthly"     : pd.Series(ten_year_minus_federal_funds_rate_monthly['Value'],index=ten_year_minus_federal_funds_rate_monthly.index),
         #   "consumer_sentiment"                            : pd.Series(consumer_sentiment['Index'],index=consumer_sentiment.index),
        }
    
    def get_coincident_indicators(self): 
        ndl = self._require_nasdaq_data_link()
        return {
            "non-farm payrolls": ndl.get("FRED/PAYEMS"),
            "aggregate_real_personal_income_ex_transfer_payments": ndl.get("FRED/PIECTR"),
            "industrial_production_index": ndl.get("FRED/INDPRO"),
            "manufacturing_and_trade_sales": ndl.get("FRED/M0602AUSM144SNBR")
        }
    
    def get_lagging_indicators(self):
        ndl = self._require_nasdaq_data_link()
        return {
            "average_duration_of_unemplymoyment": ndl.get("FRED/UEMPMEAN"),
            "inventory_sales_ratio": ndl.get("FRED/ISRATIO"),
            "change_in_unit_labor_costs": ndl.get("FRED/ULCNFB"),
            "average_bank_prime_lending_rate": ndl.get("FRED/WPRIME"),
            "commercial and industrial loans outstanding": ndl.get("FRED/BUSLOANS"),
            "consumer_installment_debt_to_income": ndl.get("FRED/TDSP"),
            "consumer_price_index_for_services": ndl.get("FRED/CUSR0000SAS")
        }
