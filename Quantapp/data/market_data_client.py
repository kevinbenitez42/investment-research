from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from Quantapp.data.repositories.market_constituents_repository import (
    get_market_constituent_data,
    get_market_index_tables,
)
from Quantapp.data.sources.yfinance_history import Ticker, download_history

class MarketDataClient:
    
    def __init__(self):
        pass
        
    def get_broad_market_data(self):
        broad_market = {
            'Global Equities': 'ACWI',
            'U.S. Total Equity': 'VTI',
            'U.S. Large Cap': 'SPY',
            'International ex-U.S.': 'VXUS',
            'Emerging Markets': 'IEMG',
            'U.S. Aggregate Bonds': 'AGG',
            'Global Bonds': 'BNDW',
            'Broad Commodities': 'DBC',
            'Gold': 'GLD',
            'Crypto (Bitcoin)': 'BTC-USD',
            'U.S. Real Estate': 'VNQ',
            'U.S. Dollar Index': 'DX-Y.NYB'
        }
        return {
            name: Ticker(ticker).history(period='max', interval='1d')
            for name, ticker in broad_market.items()
        }
    
    def get_major_equity_indices_data(self):
        return {
            'S&P 500' : Ticker('SPY').history(period='max', interval='1d'),
            'NASDAQ 100' : Ticker('QQQ').history(period='max', interval='1d'),
            'Dow Jones Industrial Average' : Ticker('DIA').history(period='max', interval='1d'),
            'Russell 2000' : Ticker('IWM').history(period='max', interval='1d')
        }
        
    def get_market_cap_data(self):
        return {
            'Large Cap' : Ticker('IVV').history(period='max', interval='1d'),
            'Mid Cap'   : Ticker('IJH').history(period='max', interval='1d'),
            'Small Cap' : Ticker('IJR').history(period='max', interval='1d')
        }
        
    def get_sector_data(self):
        return {
            'Healthcare' : Ticker('XLV').history(period='max', interval='1d'),
            'Communications' : Ticker('XLC').history(period='max', interval='1d'),
            'Technology' : Ticker('XLK').history(period='max', interval='1d'),
            'Financial' : Ticker('XLF').history(period='max', interval='1d'),
            'Industrial' : Ticker('XLI').history(period='max', interval='1d'),
            'Materials' : Ticker('XLB').history(period='max', interval='1d'),
            'Consumer Discretionary' : Ticker('XLY').history(period='max', interval='1d'),
            'Consumer Staples' : Ticker('XLP').history(period='max', interval='1d'),
            'Real Estate'      : Ticker('XLRE').history(period='max', interval='1d'),
            'Utilities'        :Ticker('XLU').history(period='max', interval='1d'),
            'Energy' : Ticker('XLE').history(period='max', interval='1d')
        }
        
    def get_commodity_data(self):
        return {
            'Agriculture' :  Ticker('DBA').history(period='max', interval='1d'),
            'Energy'      :  Ticker('DBE').history(period='max', interval='1d'),
            'Base Metals' :  Ticker('DBE').history(period='max', interval='1d'),
            'Precious Metals': Ticker('GLTR').history(period='max', interval='1d')
        }
        
    def get_international_data(self):    
        return {
             'Emerging Market' : Ticker('EEM').history(period='max', interval='1d'),
             'Frontier Markets' : Ticker('FM').history(period='max', interval='1d'),
        }
        
    def get_qualitative_factors(self):
        return {
            'Buybacks': Ticker('PKW').history(period='max', interval='1d'),
            'Spin-Offs': Ticker('CSD').history(period='max', interval='1d'),
            'Hedgefunds': Ticker('GURU').history(period='max', interval='1d'),
            'IPOs': Ticker('IPO').history(period='max', interval='1d'),
            'Mergers & Acquisitions': Ticker('MNA').history(period='max', interval='1d'),
            'Quality': Ticker('QUAL').history(period='max', interval='1d'),
            'Private Equity': Ticker('PSP').history(period='max', interval='1d'),
        }
        
    def get_factor_data(self):
        return {
            'Growth' : Ticker('SPYG').history(period='max', interval='1d'),
            'Value' : Ticker('VLUE').history(period='max', interval='1d'),
            'Momentum' : Ticker('MTUM').history(period='max', interval='1d'),
            'Quality' : Ticker('QUAL').history(period='max', interval='1d'),
            'Market Capitalization' : Ticker('SIZE').history(period='max', interval='1d'),
            'Low Volatility' : Ticker('USMV').history(period='max', interval='1d'),
            'High Dividend' : Ticker('VYM').history(period='max', interval='1d'),
        }
        
    def get_beta_factors(self):
        return {
            'Low Beta'   : Ticker('SPLV').history(period='max', interval='1d'),
            'High Beta'  : Ticker('SPHB').history(period='max', interval='1d'),
        }
    
    def get_dividend_data(self):
        return {
            'High Yield' : Ticker('VYM').history(period='max', interval='1d'),
            'Low Yield' : Ticker('DVY').history(period='max', interval='1d'),
            'Dividend Growth' : Ticker('DGRO').history(period='max', interval='1d'),
            'Dividend Value' : Ticker('SCHD').history(period='max', interval='1d'),
            'Dividend Aristocrats' : Ticker('NOBL').history(period='max', interval='1d'),
            'Dividend Achievers' : Ticker('PFM').history(period='max', interval='1d'),
            'Dividend Kings' : Ticker('KNGS').history(period='max', interval='1d'),
            'Dividend Champions' : Ticker('SDY').history(period='max', interval='1d'),
        }
   
    def get_size_vs_value_data(self):
        
        return {
            'Large Cap Value': Ticker('IVE').history(period='max', interval='1d'),
            'Large Cap Growth': Ticker('IVW').history(period='max', interval='1d'),
            'Large Cap Core': Ticker('IVV').history(period='max', interval='1d'),
            'Mid Cap Value': Ticker('IJJ').history(period='max', interval='1d'),
            'Mid Cap Growth': Ticker('IJK').history(period='max', interval='1d'),
            'Mid Cap Core': Ticker('IJH').history(period='max', interval='1d'),
            'Small Cap Value': Ticker('IJS').history(period='max', interval='1d'),
            'Small Cap Growth': Ticker('IJT').history(period='max', interval='1d'),
            'Small Cap Core': Ticker('IJS').history(period='max', interval='1d'),
        }  
        
    def get_allocation_data(self):
        return {
            'Growth' : Ticker('AOR').history(period='max', interval='1d'),
            'Moderate' : Ticker('AOM').history(period='max', interval='1d'),
            'Aggresive': Ticker('AOA').history(period='max', interval='1d'),
            'Conservative': Ticker('AOK').history(period='max', interval='1d'),
        }

    def get_bond_data(self, type='bond market'):
        if type == 'bond market':
            bond_data = {
                'Treasuries': "TLT",                  # Broad exposure to long-term US Treasuries
                'TIPS': "TIP",                        # Treasury Inflation-Protected Securities
                'Investment_Grade': "LQD",            # Broad US Investment-Grade Corporate Bonds
                'High_Yield': "HYG",                  # US High-Yield Corporate Bonds
                'Floating_Rate': "BKLN",              # US Senior Loans / Floating Rate Bonds
                'Convertibles': "CWB",                # US Convertible Bonds
                'Preferreds': "PFF",                  # US Preferred Stocks
                'Munis': "MUB",                       # Broad US Municipal Bonds
                'Agency_MBS': "MBB",                  # Agency Mortgage-Backed Securities
                'Commercial_MBS': "CMBS",             # Commercial Mortgage-Backed Securities
                'Asset_Backed': "ABS",                # Asset-Backed Securities
                'Global_Hedged': "BNDX",              # Developed Market Bonds, hedged to USD
                'Global_Unhedged': "IGOV",            # Developed Market Bonds, unhedged
                'EM_USD': "EMB",                       # Emerging Market Bonds (USD)
                'EM_Local': "EMLC"                     # Emerging Market Bonds (Local Currency)
            }
        elif type == 'US Government':
            bond_data = {
                'US Treasury': 'GOVT',
                'TIPS': 'TIP',
                'US Agencies': 'AGZ',
                'Mortgage Backed': 'MBB',
                'Commercial Mortgage': 'CMBS',
                'Ginnie Mae': 'GNMA',
                'Municipal Bonds': 'MUB',
                'Municipal Short-Term': 'TFI'
            }
        elif type == 'Treasuries':
            bond_data = {
                "US Ultra Short-Term Treasury": "SHV",
                "US Short-Term Treasury": "BIL",
                "US Treasury 1-3 Year (SGOV)": "SGOV",
                "US Treasury 1-3 Year (SHY)": "SHY",
                "US Treasury 3-7 Year": "IEF",
                'US Treasury 10-20 Year': 'TLH',
                "US Treasury 20+ Year": "TLT",
                "US Zero Coupon 25+ Year Treasury": "ZROZ",
                'Laddered Treasury': 'VGLT'
            }
        elif type == 'Corporate Bonds':
            bond_data = {
                'USD Aggregate': 'AGG',
                'Senior Loans': 'BKLN',
                'High grade Corp': 'LQD',
                'High Yield Corp': 'HYG',
                'Convertible Bonds': 'CWB',
                'Preferred Stocks': 'PFF',
                'HG Floating Corp': 'FLOT',
                'HG Short-Term Corp': 'MINT'
            }
        elif type == 'low risk credit':
            bond_data = {
                "US Ultra Short-Term Treasury": "SHV",
                "US Treasury 1-3 Year (SGOV)": "SGOV",
                "US Treasury 1-3 Year (SHY)": "SHY",
                "US Treasury 3-7 Year": "IEF",
                "US Treasury 20+ Year": "TLT"
            }
        elif type == 'medium risk credit':
            bond_data = {
                "Investment-Grade Corporate Bonds": "LQD",
                "Municipal Bonds": "MUB",
                "Agency Mortgage-Backed Securities": "MBB"
            }
        elif type == 'high risk credit':
            bond_data = {
                "High-Yield Corporate Bonds": "HYG",
                "High-Yield Corporate Bonds (JNK)": "JNK",
                "Short-Term High-Yield Bonds": "SJNK",
                "Short-Term High-Yield Bonds (SHYG)": "SHYG",
                "Senior Loans / Floating Rate": "BKLN",
                "Convertible Bonds": "CWB",
                "Emerging Market Bonds (USD)": "EMB",
                "Emerging Market Bonds (Local Currency)": "EMLC"
            }
        elif type == 'convertible bonds':
            bond_data = {
                "US Convertible Bonds": "CWB"
            }
        elif type == 'preferred stocks':
            bond_data = {
                "US Preferred Stocks (PFF)": "PFF",
                "US Preferred Stocks (PGX)": "PGX"
            }
        elif type == 'municipal bonds':
            bond_data = {
                "US Municipal Bonds (MUB)": "MUB",
                "US Short-Term Municipal Bonds (TFI)": "TFI"
            }
        elif type == 'structured credit':
            bond_data = {
                "Agency Mortgage-Backed Securities": "MBB",
                "Commercial Mortgage-Backed Securities": "CMBS",
                "Asset-Backed Securities": "ABS"
            }
        elif type == 'international/sovereign':
            bond_data = {
                "Developed Markets Global Hedged": "BNDX",
                "Developed Markets Global Unhedged": "IGOV",
                "Emerging Market Bonds (USD)": "EMB",
                "Emerging Market Bonds (Local Currency)": "EMLC"
            }
        elif type == 'floating rate':
            bond_data = {
                "S&P U.S. Dollar Denominated Leveraged Loans": "BKLN",
                "Solactive LSTA U.S. Leveraged Loans": "SRLN"
            }
        return {
            name: Ticker(ticker).history(period='max', interval='1d')
            for name, ticker in bond_data.items()
        }
    
    def get_forex_data(self, category='major pairs'):
        if category == 'major pairs':
            forex_data = {
                'EUR/USD': 'EURUSD=X',
                'GBP/USD': 'GBPUSD=X',
                'USD/JPY': 'JPY=X',
                'USD/CHF': 'CHF=X',
                'AUD/USD': 'AUDUSD=X',
                'USD/CAD': 'CAD=X',
                'NZD/USD': 'NZDUSD=X'
            }
        elif category == 'minor pairs':
            forex_data = {
                'EUR/GBP': 'EURGBP=X',
                'EUR/JPY': 'EURJPY=X',
                'GBP/JPY': 'GBPJPY=X',
                'AUD/JPY': 'AUDJPY=X',
                'CHF/JPY': 'CHFJPY=X',
                'EUR/AUD': 'EURAUD=X'
            }
        elif category == 'emerging market pairs':
            forex_data = {
                'USD/MXN': 'USDMXN=X',
                'USD/ZAR': 'USDZAR=X',
                'USD/TRY': 'USDTRY=X',
                'USD/BRL': 'USDBRL=X',
                'USD/INR': 'USDINR=X',
                'USD/SGD': 'USDSGD=X',
                'USD/HKD': 'USDHKD=X'
            }
        elif category == 'safe-haven currencies':
            forex_data = {
                'USD': 'USD=X',
                'JPY': 'JPY=X',
                'CHF': 'CHF=X'
            }
        elif category == 'commodity-linked currencies':
            forex_data = {
                'AUD': 'AUD=X',
                'NZD': 'NZD=X',
                'CAD': 'CAD=X'
            }
        elif category == 'major currencies':
            forex_data = {
                'USD': 'USD=X',
                'EUR': 'EUR=X',
                'JPY': 'JPY=X',
                'GBP': 'GBP=X',
                'CHF': 'CHF=X',
                'CAD': 'CAD=X',
                'AUD': 'AUD=X',
                'NZD': 'NZD=X'
            }
        elif category == 'emerging / exotic currencies':
            forex_data = {
                'MXN': 'MXN=X',
                'ZAR': 'ZAR=X',
                'TRY': 'TRY=X',
                'BRL': 'BRL=X',
                'INR': 'INR=X',
                'SGD': 'SGD=X',
                'HKD': 'HKD=X'
            }
        else:
            raise ValueError(f"Unknown category: {category}")

        return {
            name: Ticker(ticker).history(period='max', interval='1d')
            for name, ticker in forex_data.items()
        }
    
    def _get_broad_region_map(self):
        return {
            'Global': 'ACWI',
            'North America (U.S. proxy)': 'VTI',
            'Europe': 'IEUR',
            'Pacific ex-Japan': 'EPP',
            'Japan': 'EWJ',
            'Latin America': 'ILF',
            'Middle East (Saudi proxy)': 'KSA',
            'Africa': 'AFK',
            'Emerging Markets': 'EEM',
            'Developed Markets': 'EFA',
            'Frontier Markets': 'FM'
        }

    def get_world_data(self):
        region_data = self._get_broad_region_map()
        return {
            name: Ticker(ticker).history(period='max', interval='1d')
            for name, ticker in region_data.items()
        }

    def get_region_data(self, region):
        # Keep "Asia Pacific" as an alias so older notebooks continue to work.
        if region in ('APAC', 'Asia Pacific'):
            region_data = {
                'China': 'MCHI',
                'India': 'INDA',
                'Vietnam': 'VNM',
                'Indonesia': 'EIDO',
                'Thailand': 'THD',
                'Philippines': 'EPHE',
                'Malaysia': 'EWM',
                'Pakistan': 'PAK',
                'Sri Lanka': 'CEY',
                'Japan': 'EWJ',
                'South Korea': 'EWY',
                'Hong Kong': 'EWH',
                'Singapore': 'EWS',
                'Taiwan': 'EWT',
                'Australia': 'EWA',
                'New Zealand': 'ENZL'
            }
        elif region == 'East Asia':
            region_data = {
                'China': 'MCHI',
                'Japan': 'EWJ',
                'South Korea': 'EWY',
                'Hong Kong': 'EWH',
                'Taiwan': 'EWT'
            }
        elif region == 'Southeast Asia':
            region_data = {
                'Vietnam': 'VNM',
                'Indonesia': 'EIDO',
                'Thailand': 'THD',
                'Philippines': 'EPHE',
                'Malaysia': 'EWM',
                'Singapore': 'EWS'
            }
        elif region == 'South Asia':
            region_data = {
                'India': 'INDA',
                'Pakistan': 'PAK',
                'Sri Lanka': 'CEY'
            }
        elif region == 'Oceania':
            region_data = {
                'Australia': 'EWA',
                'New Zealand': 'ENZL'
            }
        elif region == 'Middle East':
            region_data = {
                'Turkey': 'TUR',
                'Saudi Arabia': 'KSA',
                'Israel': 'EIS',
                'Qatar': 'QAT',
                'United Arab Emirates': 'UAE',
                'Egypt': 'EGPT',
                'Jordan': 'JOR',
                'Bahrain': 'BHR',
                'Oman': 'OMN'
            }
        elif region == 'Africa':
            region_data = {
                'South Africa': 'EZA',
                'Nigeria': 'NGE',
                'Ghana': 'GHAN',
                'Kenya': 'KEN',
                'Morocco': 'MORL',
                'Africa Ex South Africa': 'AFK',
                'Zambia': 'ZMB',
                'Zimbabwe': 'ZWE'
            }
        elif region == 'Europe':
            region_data = {
                'Germany': 'EWG',
                'UK': 'EWU',
                'France': 'EWQ',
                'Italy': 'EWI',
                'Spain': 'EWP',
                'Switzerland': 'EWL',
                'Netherlands': 'EUN',
                'Sweden': 'EWD',
                'Denmark': 'EDEN',
                'Norway': 'ENOR',
                'Finland': 'EFNL',
                'Iceland': 'EICL',
                'Belgium': 'EBEL',
                'Austria': 'EWO',
                'Ireland': 'EIRL',
                'Portugal': 'PGAL',
                'Greece': 'GREK',
                'Czech Republic': 'CZE',
                'Hungary': 'HUN',
                'Poland': 'EPOL',
                'Slovakia': 'ESK'
            }
        elif region == 'Western Europe':
            region_data = {
                'Germany': 'EWG',
                'France': 'EWQ',
                'Netherlands': 'EUN',
                'Belgium': 'EBEL',
                'Austria': 'EWO',
                'Switzerland': 'EWL'
            }
        elif region == 'Eastern Europe':
            region_data = {
                'Poland': 'EPOL',
                'Czech Republic': 'CZE',
                'Hungary': 'HUN',
                'Slovakia': 'ESK'
            }
        elif region == 'Southern Europe':
            region_data = {
                'Italy': 'EWI',
                'Spain': 'EWP',
                'Portugal': 'PGAL',
                'Greece': 'GREK'
            }
        elif region == 'Northern Europe':
            region_data = {
                'UK': 'EWU',
                'Sweden': 'EWD',
                'Denmark': 'EDEN',
                'Norway': 'ENOR',
                'Finland': 'EFNL',
                'Iceland': 'EICL'
            }
        elif region == 'Ireland':
            region_data = {
                'Ireland': 'EIRL'
            }
        elif region == 'North America':
            region_data = {
                'United States': 'SPY',
                'Canada': 'EWC',
                'Mexico': 'EWW'
            }
        elif region == 'South America':
            region_data = {
                'Brazil': 'EWZ',
                'Chile': 'ECH',
                'Peru': 'EPU',
                'Argentina': 'ARGT',
                'Colombia': 'GXG'
            }
        elif region == 'Emerging Markets':
            region_data = {
                'China': 'MCHI',
                'India': 'INDA',
                'Brazil': 'EWZ',
                'Russia': 'ERUS',
                'South Africa': 'EZA',
                'Turkey': 'TUR',
                'Vietnam': 'VNM',
                'Mexico': 'EWW',
                'Indonesia': 'EIDO',
                'Thailand': 'THD',
                'Philippines': 'EPHE',
                'Malaysia': 'EWM',
                'Colombia': 'GXG',
                'Chile': 'ECH',
                'Peru': 'EPU',
                'Egypt': 'EGPT',
                'Argentina': 'ARGT',
                'Pakistan': 'PAK',
                'Nigeria': 'NGE',
                'Ghana': 'GHAN',
                'Kenya': 'KEN',
                'Morocco': 'MORL',
                'Sri Lanka': 'CEY',
                'Africa Ex South Africa': 'AFK'
            }
        elif region == 'Developed Markets':
            region_data = {
                'Japan': 'EWJ',
                'Germany': 'EWG',
                'UK': 'EWU',
                'Canada': 'EWC',
                'Australia': 'EWA',
                'France': 'EWQ',
                'Italy': 'EWI',
                'Spain': 'EWP',
                'South Korea': 'EWY',
                'Hong Kong': 'EWH',
                'Singapore': 'EWS',
                'Taiwan': 'EWT',
                'Switzerland': 'EWL',
                'Netherlands': 'EUN',
                'Sweden': 'EWD',
                'Denmark': 'EDEN',
                'Norway': 'ENOR',
                'Finland': 'EFNL',
                'Iceland': 'EICL',
                'Belgium': 'EBEL',
                'Austria': 'EWO',
                'Ireland': 'EIRL',
                'New Zealand': 'ENZL',
                'Portugal': 'PGAL',
                'Greece': 'GREK',
                'Czech Republic': 'CZE',
                'Hungary': 'HUN',
                'Poland': 'EPOL',
                'Slovakia': 'ESK'
            }
        elif region == 'Frontier Markets':
            region_data = {
                'Vietnam': 'VNM',
                'Pakistan': 'PAK',
                'Nigeria': 'NGE',
                'Africa Ex South Africa': 'AFK'
            }
        elif region == 'Broad':
            region_data = self._get_broad_region_map()

        return {
            name: Ticker(ticker).history(period='max', interval='1d')
            for name, ticker in region_data.items()
        }
        
    def get_volatility_data(self):
        return {
            'VIX (1 Month)': Ticker('VIX').history(period='max', interval='1d'),
            'VIX (6 Month)': Ticker('VIXM').history(period='max', interval='1d'),
            'SKEW': Ticker('^SKEW').history(period='max', interval='1d'),
            'MOVE': Ticker('^MOVE').history(period='max', interval='1d'),
        }
    
    def get_strategy_data(self):
        return {
            'Active Investing': Ticker('QAI').history(period='max', interval='1d'),
            'Beta Rotation': Ticker('BTAL').history(period='max', interval='1d'),
            'Covered Calls': Ticker('PBP').history(period='max', interval='1d'),
            'Hedged'       : Ticker('PHDG').history(period='max', interval='1d')
        }
                  
    def get_market_assets(self):
        # Get the market tables
        tables = self.retrieve_market_tables()

        sp500_table = tables["SP500_TABLE"]
    #    qqq_table = tables["NASDAQ_100_TABLE"]
        dia_table = tables["DIA_TABLE"]
        #russell_1000_table = tables["Russell_1000_TABLE"]

        # Retrieve all companies from each sector
        xlk_table = sp500_table[sp500_table['Sector'] == 'Information Technology']
        xlf_table = sp500_table[sp500_table['Sector'] == 'Financials']
        xlv_table = sp500_table[sp500_table['Sector'] == 'Health Care']
        xli_table = sp500_table[sp500_table['Sector'] == 'Industrials']
        xly_table = sp500_table[sp500_table['Sector'] == 'Consumer Discretionary']
        xle_table = sp500_table[sp500_table['Sector'] == 'Energy']
        xlb_table = sp500_table[sp500_table['Sector'] == 'Materials']
        xlc_table = sp500_table[sp500_table['Sector'] == 'Communication Services']
        xlre_table = sp500_table[sp500_table['Sector'] == 'Real Estate']
        xlp_table = sp500_table[sp500_table['Sector'] == 'Consumer Staples']
        xlu_table = sp500_table[sp500_table['Sector'] == 'Utilities']

        market_assets = {
            #ADD MIDCAP AND SMALL CAP AND EQUAL WEIGHTED ETFs
            "INDICES": ['SPY', 'QQQ', 'DIA', 'IWM','RSP','MDY','IJR'],
            "CORE_MACRO": ['ACWI', 'VTI', 'SPY', 'VXUS', 'IEMG', 'AGG', 'BNDW', 'DBC', 'GLD', 'BTC-USD', 'VNQ', 'DX-Y.NYB'],
            "SECTORS": ['XLF', 'XLK', 'XLV', 'XLC', 'XLI', 'XLU', 'XLB', 'VNQ', 'XLP', 'XLY', 'XBI', 'XLE'],
            "INDUSTRIES": ['SPY', 'SMH', 'KRE', 'KIE', 'KBE', 'IAK','JETS','XHB','ITA','IGV'],
            "MID_CAP_SECTORS": ['IJH', 'IJK', 'IJH', 'IYF', 'IYH', 'IYZ', 'IYT', 'IYW', 'IYR', 'IYC', 'IYJ', 'IYE'],
            "SPY_HOLDINGS": sp500_table['Symbol'].tolist(),
        #    "QQQ_HOLDINGS": qqq_table['Symbol'].tolist(),
            "DIA_HOLDINGS": dia_table['Symbol'].tolist(),
            # "RUSSELL_1000_HOLDINGS": russell_1000_table['Symbol'].tolist(),
            "XLK_HOLDINGS": xlk_table['Symbol'].tolist(),
            "XLF_HOLDINGS": xlf_table['Symbol'].tolist(),
            "XLI_HOLDINGS": xli_table['Symbol'].tolist(),
            "XLV_HOLDINGS": xlv_table['Symbol'].tolist(),
            "XLU_HOLDINGS": xlu_table['Symbol'].tolist(),
            "XLB_HOLDINGS": xlb_table['Symbol'].tolist(),
            "XLY_HOLDINGS": xly_table['Symbol'].tolist(),
            "XLRE_HOLDINGS": xlre_table['Symbol'].tolist(),
            "XLC_HOLDINGS": xlc_table['Symbol'].tolist(),
            "XLE_HOLDINGS": xle_table['Symbol'].tolist(),
            "XLP_HOLDINGS": xlp_table['Symbol'].tolist(),
            "BONDS": ['AGG', 'IEF', 'TLT', 'HYG', 'LQD', 'BKLN','SHY','BIL','COMT','MUB','TIP','JNK','EMB'],# 'TIPS' EXCLUDED
            "AGRICULTURE": ['DBA', 'CORN', 'WEAT', 'SOYB','WOOD'],
            "PRECIOUS_METALS": ['GLD', 'SLV', 'GDX', 'XME','NUGT','GDXJ'],
            "INDUSTRIAL_METALS": ['COPX','DBB','LIT','REMX'],
            "CRYPTO": ['BTC-USD', 'ETH-USD', 'LTC-USD', 'ADA-USD', 'SOL-USD'],
            "ENERGY": ['USO', 'UNG', 'OIH', 'XOP', 'ICLN', 'URA', 'URNM', 'GUSH', 'KOLD'],
            "FOREIGN_MARKETS": [
                'EWZ', 'EWJ', 'EWA', 'EWG', 'EWW', 'EEM', 'EFA', 'FEZ', 'INDA', 'EWU',
                'EWG', 'EWL', 'XIU', 'VAS', 'ENZL', 'TUR', 'EZA', 'EWS', 'EWH','EWT','EWY','FXI'
            ],
            "PRIMARY_SECTORS": ['USO', 'GLD', 'SPY', 'VNQ', 'GBTC', 'EFA', 'TLT','AGG'],
            "FX": ['UUP', 'FXE', 'FXY', 'FXA', 'FXC', 'FXB', 'INR=X', 'BRL=X', 'CNY=X'],
            "MAJOR_CURRENCY_PAIRS": ['EURUSD=X', 'JPY=X', 'GBPUSD=X', 'CHF=X', 'CAD=X', 'AUDUSD=X', 'NZDUSD=X'],
            "MINOR_CURRENCY_PAIRS": ['EURGBP=X', 'EURJPY=X', 'GBPJPY=X', 'AUDJPY=X', 'GBPCHF=X', 'AUDNZD=X'],
            "EXOTIC_CURRENCY_PAIRS": ['TRY=X', 'ZAR=X', 'SGD=X', 'HKD=X', 'MXN=X'],
            "CROSS_CURRENCY_PAIRS": ['EURCHF=X', 'EURAUD=X', 'GBPAUD=X', 'CHFJPY=X'],
            "CAPITALIZATIONS": ['SPY', 'IJH', 'IJR'],
            "INNOVATION": ['ARKG', 'ARKF', 'ARKK','CIBR','ROBO','TAN','IDRV','CLOU'],
            "ALTERNATIVE_MANAGERS": ['BX', 'KKR', 'CG', 'AEO', 'APO', 'GLPI'],
            "INFRASTRUCTURE": ['IFRA','IGF','TOLZ','PAVE','NFRA'],
            "SHIPPING": ['SEA','BDRY'],
            "LONG_LEVERAGE": ['TQQQ', 'SOXL', 'SPXL', 'TNA', 'BOIL', 'NUGT', 'ERX', 'DPST'],
            "SHORT_LEVERAGE": ['SQQQ', 'SPXS', 'UDOW', 'SSO', 'TECL', 'FAS', 'NVDA', 'TQQQ', 'VXX', 'UVXY', 'VIXY', 'UVIX', 'SVXY', 'SOXS', 'TZA', 'USD', 'TSLL', 'LABU', 'DPST', 'NUGT', 'CONL'],
            "SINGLE_FACTOR": ['QUAL', 'VLUE', 'MTUM', 'SIZE', 'USMV'],
            "MULTI_FACTOR": ['LRGF', 'INTF', 'GLOF'],
            "MINIMUM_VOLATILITY": ['USMV', 'EFAV', 'EEMV'],
        }

        return market_assets
    def retrieve_market_tables(self):
        return get_market_index_tables()
    
    def retrieve_market_data(self):
        return get_market_constituent_data()
    
    def generate_series(self,tickers, columns=['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'], period='10y', interval='1d'):
        """
        Generate a DataFrame or Series containing the specified columns for the given tickers.

        Parameters:
        - tickers: List of ticker symbols or a single ticker symbol.
        - columns: List of columns to retrieve or a single column to retrieve (default is ['Close']).
        - period: Data period to retrieve (default is '1y').
        - interval: Data interval to retrieve (default is '1d').

        Returns:
        - pd.DataFrame or pd.Series with the specified columns for the given tickers.
        """
        # Ensure tickers and columns are lists
        if isinstance(tickers, str):
            tickers = [tickers]
        if isinstance(columns, str):
            columns = [columns]

        tickers = [ticker.replace('.', '-') for ticker in tickers]
        try:
            df = download_history(tickers, period=period, interval=interval, progress=False)
        except Exception as e:
            print(f"An error occurred while fetching data: {e}")
            return pd.DataFrame()
        
        # Check if the specified columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns.get_level_values(0)]
        if missing_columns:
            print(f"Error: The following columns are not available: {missing_columns}")
            print(f"Possible columns are: {df.columns.get_level_values(0).unique().tolist()}")
            return pd.DataFrame()
        
        df = df[columns]
        
        # Handle the case where there is only one ticker and one column
        if len(tickers) == 1 and len(columns) == 1:
            return df[columns[0]].rename(tickers[0].replace('-', '.'))
        
        # Handle the case where there is only one ticker
        if len(tickers) == 1:
            df.columns = [col.replace('-', '.') for col in df.columns]
        else:
            # If only one column is selected, return a DataFrame with tickers as column names
            if len(columns) == 1:
                df = df[columns[0]]
                df.columns = [col.replace('-', '.') for col in df.columns]
            else:
                # Flatten the multi-level columns if multiple tickers are requested
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = pd.MultiIndex.from_tuples([(col[1], col[0]) for col in df.columns.values])
                else:
                    df.columns = pd.MultiIndex.from_tuples([(col.split('.')[0], col.split('.')[1]) for col in df.columns])
        
        return df
    
    def get_sector_info(ticker):
        try:
            stock = Ticker(ticker)
            sector = stock.info.get('sector', 'N/A')
            sub_industry = stock.info.get('industry', 'N/A')
            return {'Ticker': ticker, 'Sector': sector, 'Sub-Industry': sub_industry}
        except Exception as e:
            #print(f"Error fetching data for {ticker}: {e}")
            return {'Ticker': ticker, 'Sector': 'N/A', 'Sub-Industry': 'N/A'}
    
    def fetch_ticker_info(self, ticker):
        info = self.get_sector_info(ticker)
        print(info)
        print(Ticker(ticker).info)
        #market_cap = Ticker(ticker).info.get('marketCap')
        #return info['Sector'], info['Sub-Industry'], market_cap

    def get_market_caps(self,table):
        #print("Starting market cap retrieval process...")
        
        tickers = table['Symbol'].tolist()
        #print(f"Original tickers: {tickers[:10]}...")  # Print first 10 for brevity

        # Optimize ticker adjustment
        tickers = ['BRK-B' if symbol == 'BRK.B' else 'BF-B' if symbol == 'BF.B' else symbol for symbol in tickers]
        #print(f"Adjusted tickers: {tickers[:10]}...")  # Print first 10 for brevity

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.fetch_ticker_info, tickers))

        # Unpack results
        sectors, sub_industries, market_caps = zip(*results)

        table['Sector'] = sectors
        table['Sub-Industry'] = sub_industries
        table['Market Cap'] = market_caps
        
        #print("Market cap retrieval process completed.")
        return table
    
    def get_market_cap_threshold_companies(self,info):
        """
        Calculates market cap rankings and identifies companies contributing to specified cumulative market cap thresholds.

        Parameters:
            info (pd.DataFrame): DataFrame containing at least 'Symbol' and 'Market Cap' columns.

        Returns:
            dict: A dictionary where keys are threshold labels (e.g., 'Top 50%') and values are lists of company dictionaries
                containing 'Symbol', 'Market Cap', 'Market Cap %', 'Cumulative Market Cap %', and 'Rank'.
        """
        # Step 1: Create a DataFrame of market caps
        market_caps = pd.DataFrame(info[['Symbol', 'Market Cap']])
        
        # Step 2: Sort companies by market cap in descending order
        market_caps = market_caps.sort_values(by='Market Cap', ascending=False).reset_index(drop=True)

        # Step 3: Calculate total market cap
        total_market_cap = market_caps['Market Cap'].sum()

        # Step 4: Calculate individual Market Cap %
        market_caps['Market Cap %'] = (market_caps['Market Cap'] / total_market_cap) * 100

        # Step 5: Calculate cumulative Market Cap %
        market_caps['Cumulative Market Cap %'] = market_caps['Market Cap %'].cumsum()

        # Step 6: Assign Rank
        market_caps['Rank'] = market_caps.index + 1

        # Step 7: Define thresholds
        thresholds = [50, 80]  # You can adjust or add more thresholds as needed
        threshold_dict = {}

        for threshold in thresholds:
            # Find the first index where cumulative market cap meets or exceeds the threshold
            idx = market_caps[market_caps['Cumulative Market Cap %'] >= threshold].index[0]

            # Select companies up to that index
            companies = market_caps.loc[:idx, ['Symbol', 'Market Cap', 'Market Cap %', 'Cumulative Market Cap %', 'Rank']]

            # Convert to list of dictionaries
            companies_list = companies.to_dict('records')

            # Add to the threshold dictionary with appropriate key
            threshold_key = f'Top {threshold}%'
            threshold_dict[threshold_key] = companies_list

        return threshold_dict
