import os

import pandas as pd
import requests
import yfinance as yf

from Quantapp.data.company_data_client import CompanyDataClient

class GICSDataClient:
    
    def __init__(self, client=None,save_path=None, debug=False):
        self.client = client
        self.save_path = save_path if save_path is not None else os.getcwd()
        self.debug = debug
    
    def _log(self, message):
        if self.debug:
            print(message)
    
    def get_latest_gics_structure(self):
        url = "https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard"

        # Spoof browser headers to avoid 403
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }

        # Fetch the page
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        html = response.text

        # Parse tables from HTML
        tables = pd.read_html(html)
        gics_table = tables[0]  # the first table contains the GICS structure

        # Set column names for the GICS table
        gics_table.columns = ['Sector Code', 'Sector Name', 'Industry Group Code', 'Industry Group Name',
                            'Industry Code', 'Industry Name', 'Sub-Industry Code', 'Sub-Industry Name']

        # Convert code columns to integers, handling any non-numeric values
        gics_table['Sector Code'] = pd.to_numeric(gics_table['Sector Code'], errors='coerce').astype('Int64')
        gics_table['Industry Group Code'] = pd.to_numeric(gics_table['Industry Group Code'], errors='coerce').astype('Int64')
        gics_table['Industry Code'] = pd.to_numeric(gics_table['Industry Code'], errors='coerce').astype('Int64')
        gics_table['Sub-Industry Code'] = pd.to_numeric(gics_table['Sub-Industry Code'], errors='coerce').astype('Int64')

        # Drop rows with NaN values and reset the index
        gics_table = gics_table.dropna().reset_index(drop=True)
        
        # Save table to current directory as gics_structure.csv
        gics_table.to_csv(os.path.join(self.save_path, 'gics_structure.csv'), index=False)
        
        return gics_table
    
    # Retrieve all companies from S&P 500, S&P 400, S&P 600 with their GICS codes and capitalization
    def retrieve_companies(self):
        # URLs for the market data
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp400_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        sp600_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        sp500_response = requests.get(sp500_url, headers=headers)
        sp400_response = requests.get(sp400_url, headers=headers)
        sp600_response = requests.get(sp600_url, headers=headers)
        
        
        # Retrieve S&P 500 data KEEP THIS THIS SHIFT THIS SHIT ARROUND TOO MUCH
        #--------------------------------------------------------------------------------------------
        #sp500_table = pd.read_html(response.text)[0] 
        
        sp500_table = pd.read_html(sp500_response.text)[0] 
        sp400_table = pd.read_html(sp400_response.text)[0]
        sp600_table = pd.read_html(sp600_response.text)[0]
        
        #--------------------------------------------------------------------------------------------
        sp500_table = sp500_table[['Symbol', 'GICS Sector', 'GICS Sub-Industry']]
        sp500_table = sp500_table.rename(columns={'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'})
        
        sp400_table = sp400_table[['Symbol', 'GICS Sector', 'GICS Sub-Industry']]
        sp400_table = sp400_table.rename(columns={'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'})
        
        sp600_table = sp600_table[['Symbol', 'GICS Sector', 'GICS Sub-Industry']]
        sp600_table = sp600_table.rename(columns={'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Sub-Industry'})
        
        
        sp600_table['Capitalization'] = 'Small Cap'
        sp400_table['Capitalization'] = 'Mid Cap'
        sp500_table['Capitalization'] = 'Large Cap'
        
    
        # Combine all tables into one DataFrame
        combined_table = pd.concat([sp500_table, sp400_table, sp600_table], ignore_index=True)
        #add a column called GICS Code, find the GICS code for each sub-industry using the name_to_gics function (make sure its an int, currently its float)
        combined_table['GICS Code'] = combined_table['Sub-Industry'].apply(lambda x: self.name_to_gics(x, level='Sub-Industry'))
        combined_table['GICS Code'] = combined_table['GICS Code'].astype('Int64')
        
        gics_table = pd.read_csv(os.path.join(self.save_path, 'gics_structure.csv'))
        
        #these are the unique GICS codes in the combined table
        all_gics_codes_combined_table = combined_table['GICS Code'].unique().tolist()
        #for each of these gics codes, retrieve 'Sector', 'Industry Group', 'Industry', 'Sector code', 'Industry Group Code', 'Industry Code',  from gics_table
        #add thess columns to the combined table
        gics_info_list = []
        for gics_code in all_gics_codes_combined_table:
            gics_info = gics_table[gics_table['Sub-Industry Code'] == gics_code]
            if not gics_info.empty:
                gics_info_list.append({
                    'GICS Code': gics_code,
                    'Sector': gics_info.iloc[0]['Sector Name'],
                    'Industry Group': gics_info.iloc[0]['Industry Group Name'],
                    'Industry': gics_info.iloc[0]['Industry Name'],
                    'Sector Code': gics_info.iloc[0]['Sector Code'],
                    'Industry Group Code': gics_info.iloc[0]['Industry Group Code'],
                    'Industry Code': gics_info.iloc[0]['Industry Code']
                })
                
        gics_info_df = pd.DataFrame(gics_info_list)
      
        combined_table = pd.merge(combined_table, gics_info_df, on='GICS Code', how='left')
        combined_table.rename(columns={
            'Sector_x': 'Sector',
        }, inplace=True)
        combined_table = combined_table.drop(columns=['Sector_y'])
        #make sure the Industry Group code, Industry Code, Sector Code are Int64
        combined_table['Sector Code'] = combined_table['Sector Code'].astype('Int64')
        combined_table['Industry Group Code'] = combined_table['Industry Group Code'].astype('Int64')
        combined_table['Industry Code'] = combined_table['Industry Code'].astype('Int64')
        
        #REORDER COLUMNS
        combined_table = combined_table[['Symbol', 'Capitalization', 'Sector','Sector Code', 'Industry Group', 'Industry Group Code', 'Industry', 'Industry Code', 'Sub-Industry', 'GICS Code']]

        return combined_table
  
    # Convert GICS code to names (return dictionary with sector, industry group, industry, sub-industry)
    def gics_to_name(self, gics_code):
        gics_table = pd.read_csv(os.path.join(self.save_path, 'gics_structure.csv'))
        
        gics_code_str = str(gics_code)
        code_length = len(gics_code_str)
        
        if code_length == 2:
            sector_row = gics_table[gics_table['Sector Code'] == int(gics_code_str)]
            if not sector_row.empty:
                return {
                    "Sector": sector_row.iloc[0]['Sector Name']
                }
        elif code_length == 4:
            industry_group_row = gics_table[gics_table['Industry Group Code'] == int(gics_code_str)]
            if not industry_group_row.empty:
                return {
                    "Sector": industry_group_row.iloc[0]['Sector Name'],
                    "Industry Group": industry_group_row.iloc[0]['Industry Group Name']
                }
        elif code_length == 6:
            industry_row = gics_table[gics_table['Industry Code'] == int(gics_code_str)]
            if not industry_row.empty:
                return {
                    "Sector": industry_row.iloc[0]['Sector Name'],
                    "Industry Group": industry_row.iloc[0]['Industry Group Name'],
                    "Industry": industry_row.iloc[0]['Industry Name']
                }
        elif code_length == 8:
            sub_industry_row = gics_table[gics_table['Sub-Industry Code'] == int(gics_code_str)]
            if not sub_industry_row.empty:
                return {
                    "Sector": sub_industry_row.iloc[0]['Sector Name'],
                    "Industry Group": sub_industry_row.iloc[0]['Industry Group Name'],
                    "Industry": sub_industry_row.iloc[0]['Industry Name'],
                    "Sub-Industry": sub_industry_row.iloc[0]['Sub-Industry Name']
                }
        return None
    
    # Convert GICS names to codes (return GICS code based on level: Sector, Industry Group, Industry, Sub-Industry)
    def name_to_gics(self, name, level='Sector'):
        gics_table = pd.read_csv(os.path.join(self.save_path, 'gics_structure.csv'))
        if level == 'Sector':
            row = gics_table[gics_table['Sector Name'] == name]
            if not row.empty:
                return row.iloc[0]['Sector Code']
        elif level == 'Industry Group':
            row = gics_table[gics_table['Industry Group Name'] == name]
            if not row.empty:
                return row.iloc[0]['Industry Group Code']
        elif level == 'Industry':
            row = gics_table[gics_table['Industry Name'] == name]
            if not row.empty:
                return row.iloc[0]['Industry Code']
        elif level == 'Sub-Industry':
            row = gics_table[gics_table['Sub-Industry Name'] == name]
            if not row.empty:
                return row.iloc[0]['Sub-Industry Code']
        return None
    
    # retrieves all child GICS codes given a parent code
    def retrieve_children(self, parent_code):
        gics_table = pd.read_csv(os.path.join(self.save_path, 'gics_structure.csv'))
        
        parent_code_str = str(parent_code)
        code_length = len(parent_code_str)
        children_set = set()
        
        if code_length == 2:
            industry_group_rows = gics_table[gics_table['Sector Code'] == int(parent_code_str)]
            for _, row in industry_group_rows.iterrows():
                children_set.add((row['Industry Group Code'], row['Industry Group Name'], 'Industry Group'))
        elif code_length == 4:
            industry_rows = gics_table[gics_table['Industry Group Code'] == int(parent_code_str)]
            for _, row in industry_rows.iterrows():
                children_set.add((row['Industry Code'], row['Industry Name'], 'Industry'))
        elif code_length == 6:
            sub_industry_rows = gics_table[gics_table['Industry Code'] == int(parent_code_str)]
            for _, row in sub_industry_rows.iterrows():
                children_set.add((row['Sub-Industry Code'], row['Sub-Industry Name'], 'Sub-Industry'))
        
        children = []
        for code, name, level in children_set:
            # Recursively retrieve children
            child_dict = {
                'code': code,
                'name': name,
                'level': level,
                'children': self.retrieve_children(code)  # recursion
            }
            children.append(child_dict)
        
        return children

    #directly off of gics_table,
    def retrieve_subindustries_gic_codes(self, industry_code):
        gics_table = pd.read_csv(os.path.join(self.save_path, 'gics_structure.csv'))
        companies = self.filter_companies_by_gics(gics_code=industry_code)
        return companies['GICS Code'].tolist()
        
    #return parent as a single dictionary with all the information 
    def retrieve_parent(self, child_code):
        gics_table = pd.read_csv(os.path.join(self.save_path, 'gics_structure.csv'))
        
        child_code_str = str(child_code)
        code_length = len(child_code_str)
        
        if code_length == 4:
            row = gics_table[gics_table['Industry Group Code'] == int(child_code_str)]
            if not row.empty:
                return {
                    "Sector Code": row.iloc[0]['Sector Code'],
                    "Sector Name": row.iloc[0]['Sector Name']
                }
        elif code_length == 6:
            row = gics_table[gics_table['Industry Code'] == int(child_code_str)]
            if not row.empty:
                return {
                    "Sector Code": row.iloc[0]['Sector Code'],
                    "Sector Name": row.iloc[0]['Sector Name'],
                    "Industry Group Code": row.iloc[0]['Industry Group Code'],
                    "Industry Group Name": row.iloc[0]['Industry Group Name']
                }
        elif code_length == 8:
            row = gics_table[gics_table['Sub-Industry Code'] == int(child_code_str)]
            if not row.empty:
                return {
                    "Sector Code": row.iloc[0]['Sector Code'],
                    "Sector Name": row.iloc[0]['Sector Name'],
                    "Industry Group Code": row.iloc[0]['Industry Group Code'],
                    "Industry Group Name": row.iloc[0]['Industry Group Name'],
                    "Industry Code": row.iloc[0]['Industry Code'],
                    "Industry Name": row.iloc[0]['Industry Name']
                }
        return None
    
    # Filter companies by GICS code (could be 2,4,6,8 digits) and capitalization
    def filter_companies_by_gics(self, gics_code=None, capitalization=None):
        companies_table = self.retrieve_companies()
        
        filtered_companies = companies_table
        
        if gics_code is not None:
            gics_info = self.gics_to_name(gics_code)
            
            if gics_info is None:
                print(f"Invalid GICS code: {gics_code}")
                return pd.DataFrame()
            
            #sub_industry_gics_codes = self.retrieve_subindustries_gic_codes(gics_code)
            #display(sub_industry_gics_codes)
            # Filter companies based on the level of the provided GICS code
            code_length = len(str(gics_code))
            sector_name = gics_info['Sector']
            
            if code_length == 2:
                
                filtered_companies = companies_table[companies_table['Sector'] == sector_name]
            elif code_length == 4:
                industry_group_name = gics_info['Industry Group']
                filtered_companies = companies_table[companies_table['Sector'] == sector_name]
                filtered_companies = filtered_companies[filtered_companies['Industry Group'] == industry_group_name]
                '''                
                filtered_companies = filtered_companies[filtered_companies['Sub-Industry'].isin(
                    companies_table[companies_table['Sector'] == gics_info['Sector']]['Sub-Industry']
                )]
                '''
            elif code_length == 6:
                industry_name = gics_info['Industry']
                
                filtered_companies = companies_table[companies_table['Sector'] == sector_name]
                filtered_companies = filtered_companies[filtered_companies['Industry'] == industry_name]
                '''
                filtered_companies = filtered_companies[filtered_companies['Sub-Industry'].isin(
                    companies_table[companies_table['Sub-Industry'] == industry_name]['Sub-Industry']
                )]
                '''
            elif code_length == 8:
                
                filtered_companies = companies_table[companies_table['Sector'] == sector_name]
                filtered_companies = filtered_companies[filtered_companies['Sub-Industry'] == gics_info['Sub-Industry']]
                '''
                sub_industry_name = gics_info['Sub-Industry']
                filtered_companies = filtered_companies[filtered_companies['Sub-Industry'] == sub_industry_name]
                '''
            else:
                print(f"Invalid GICS code length: {gics_code}")
                return pd.DataFrame()
        
        
        # Further filter by capitalization if specified, fields can be 'Large Cap', 'Mid Cap', 'Small Cap'
        if capitalization is not None:
            filtered_companies = filtered_companies[filtered_companies['Capitalization'] == capitalization]
        
        return filtered_companies
    
    # Retrieve historical prices for companies in a given GICS code and capitalization
    def retrieve_prices(self, gics_code, capitalization='Large Cap', price_field='Close', period='10y', interval='1d'):
        filtered_companies = self.filter_companies_by_gics(gics_code, capitalization)
        
        if filtered_companies.empty:
            return pd.DataFrame()
        
        tickers = filtered_companies['Symbol'].tolist()
        
        # Fetch price data for the filtered tickers
        price_data = yf.Tickers(tickers).history(period=period, interval=interval)[price_field]
        
        #make sure captialization is filtered properly
        # Filter columns to include only the tickers in the filtered_companies
        price_data = price_data[[ticker for ticker in tickers if ticker in price_data.columns]]
        
        return price_data
    
    # Retrieve historical market capitalization for companies in a given GICS code and capitalization in the form of a dictionary
    def retrieve_market_cap(self, gics_code, capitalization='Large Cap', period='10y', interval='1d',should_update=False):
        #retreive companys using retrieve_companies
        companies_table = self.filter_companies_by_gics(gics_code, capitalization)
        #create a list of symbols
        symbols = companies_table['Symbol'].tolist()
        
        #retrieve market cap for each company using CompanyDataClient
        #each company has a different 'period_end_date', so we will need to align them later
        
        market_cap_data = {}
        for symbol in symbols:
            companyData = CompanyDataClient(ticker_str=symbol,client=self.client, save_path=self.save_path)
            #print(f'Retrieving market cap data for {symbol}\n')
            market_cap_data[symbol] = companyData.retrieve_market_cap(data_type='quarterly',should_update=should_update)
            
        #iterate through the dictionary and combine them into a single dataframe
        # here is the prblem, each company has a different values for 'period_end_date', we need to align them
        #combine all dataframes into a single dataframe with multi-index (period_end_date, symbol)
        combined_market_cap = pd.DataFrame()
        for symbol, df in market_cap_data.items():
            df = df.set_index('period_end_date')
            df = df.rename(columns={'market_cap': symbol})
            if combined_market_cap.empty:
                combined_market_cap = df[[symbol]]
            else:
                combined_market_cap = combined_market_cap.join(df[[symbol]], how='outer')
        #make sure to forward fill missing values
        combined_market_cap = combined_market_cap.sort_index().ffill()
        #backfill any remaining missing values
        combined_market_cap = combined_market_cap.bfill()
        return(combined_market_cap)    
    
    #simply retrieve fundamental data for companies in a given GICS code and capitalization
    def retrieve_fundamental_data(self, 
                                  gics_code, 
                                  capitalization='Large Cap', 
                                  data_type='quarterly', 
                                  statement_type='misc', 
                                  metric='pe_ratio',
                                  aggregation_method='median',
                                  filtering_method='quantile',
                                  truncate_below_zero=False,
                                  should_update=False):
        
        # Filter companies by GICS code and capitalization
        filtered_companies = self.filter_companies_by_gics(gics_code, capitalization)
        
        if filtered_companies.empty:
            return pd.DataFrame()
        
        symbols = filtered_companies['Symbol'].tolist()
        
        # Retrieve fundamental data for each company
        metric_dict = {}
        fundamental_data = {}
        for symbol in symbols:
            companyData = CompanyDataClient(ticker_str=symbol, client=self.client, save_path=self.save_path)
            # Retrieve misc data
            fundamental_data = companyData.retrieve_data(data_type=data_type, statement_type=statement_type, should_update=should_update)
            # Ensure 'period_end_date' is datetime
            fundamental_data['period_end_date'] = pd.to_datetime(fundamental_data['period_end_date'])
            # Set index to 'period_end_date'
            fundamental_data = fundamental_data.set_index('period_end_date')
            # Store the specified metric
            
            if metric in fundamental_data.columns:
                metric_dict[symbol] = fundamental_data[[metric]]
            else:
                print(f"Metric '{metric}' not found for {symbol}. Skipping.")
    

            
       
        # Combine all metric DataFrames into a single DataFrame
        combined_fundamental_data = pd.DataFrame()
        

        for symbol, df in metric_dict.items():
            df = df.rename(columns={metric: symbol})
            if combined_fundamental_data.empty:
                combined_fundamental_data = df
            else:
                combined_fundamental_data = combined_fundamental_data.join(df, how='outer')
                
        if filtering_method == 'quantile':
            # Remove outliers based on quantiles
            lower_quantile = combined_fundamental_data.quantile(0.02)
            upper_quantile = combined_fundamental_data.quantile(0.98)
            combined_fundamental_data = combined_fundamental_data[(combined_fundamental_data >= lower_quantile) & (combined_fundamental_data <= upper_quantile)]
            # Forward fill and backfill after filtering
            combined_fundamental_data = combined_fundamental_data.sort_index().ffill().bfill()
        
        if truncate_below_zero:
            combined_fundamental_data = combined_fundamental_data[combined_fundamental_data >= 0]
            # Forward fill and backfill after truncating
            combined_fundamental_data = combined_fundamental_data.sort_index().ffill().bfill()
                          
        if not combined_fundamental_data.empty:
            if aggregation_method == 'median':
                combined_fundamental_data = combined_fundamental_data.median(axis=1)
            elif aggregation_method == 'sum':
                combined_fundamental_data = combined_fundamental_data.sum(axis=1)
            elif aggregation_method == 'mean':
                combined_fundamental_data = combined_fundamental_data.mean(axis=1)
            elif aggregation_method == 'market_cap_weighted':
                market_caps = self.retrieve_market_cap(gics_code, capitalization, period='10y', interval='1d', should_update=should_update)
                market_caps = market_caps.reindex(combined_fundamental_data.index).ffill().bfill()
                weights = market_caps.div(market_caps.sum(axis=1), axis=0)
                combined_fundamental_data = (combined_fundamental_data * weights).sum(axis=1)
                self._log(combined_fundamental_data)
                
            # Add other aggregation methods if needed
        #forward fill any missing values
        combined_fundamental_data = combined_fundamental_data.sort_index().ffill()
        #backfill any remaining missing values
        combined_fundamental_data = combined_fundamental_data.bfill()
        
        combined_fundamental_data.name = f'{metric}'
        
   
        
        return combined_fundamental_data
    
    def retrieve_fundamental_data_children(self, 
                                           parent_gics_code, 
                                           capitalization='Large Cap', 
                                           as_weights=False,
                                           data_type='quarterly', 
                                           statement_type='misc', 
                                           metric='pe_ratio', 
                                           aggregation_method='median', 
                                           filtering_method='quantile', 
                                           truncate_below_zero=False, 
                                           should_update=False):
        if len(str(parent_gics_code)) in [2, 4, 6]:
            children = self.retrieve_children(parent_gics_code)
            gics_codes = [child['code'] for child in children]
            return_value = {}
            for code in gics_codes:
                self._log(f'Retrieving fundamental data for GICS Code: {code}')
                fundamental_data = self.retrieve_fundamental_data(code, capitalization, data_type, statement_type, metric, aggregation_method, filtering_method, truncate_below_zero, should_update)
                #print(f'Fundamental Data for GICS Code: {code}')
                #display(fundamental_data)
                return_value[code] = fundamental_data
                

            return_value = pd.DataFrame(return_value)
            #backfill or forward fill any missing values depending on the date
            return_value = return_value.sort_index().ffill().bfill()
            #dont aggregate, just return the dataframe with each column as the fundamental data for that gics code
            #print('Aggregated Fundamental Data for Children GICS Codes:')
            #display(return_value)
            if as_weights:
                p = return_value.div(return_value.sum(axis=1), axis=0)
                return p
            else:
                return return_value
        
    
        
        else:
            print('Parent GICS code must be 2, 4, or 6 digits long to have children.')
            return pd.DataFrame()
        
    def retrieve_market_cap_weights(self,gics_code, capitalization='Large Cap', period='10y', interval='1d',should_update=False):
        market_cap_weights = self.retrieve_market_cap(gics_code, capitalization, period, interval, should_update)
        return market_cap_weights.div(market_cap_weights.sum(axis=1), axis=0)
    
    # gics code may be 2,4,6,8 digits, retrieve all child gics codes and calculate aggregated market caps
    def retrieve_market_cap_children(self, parent_gics_code, capitalization='Large Cap', period='10y', interval='1d', should_update=False):
        if len(str(parent_gics_code)) in [2, 4, 6]:
            children = self.retrieve_children(parent_gics_code)
            gics_codes = [child['code'] for child in children]
            return_value = {}
            for code in gics_codes:
                market_cap_weights = self.retrieve_market_cap(code, capitalization, period, interval, should_update)
                total_market_cap_weights = market_cap_weights.sum(axis=1)
                #print(f'Total Market Cap Data for GICS Code: {code}')
                #display(total_market_cap_weights)
                return_value[code] = total_market_cap_weights
                

            return_value = pd.DataFrame(return_value)
            #backfill or forward fill any missing values depending on the date
            return_value = return_value.sort_index().ffill().bfill()
            #print('Aggregated Market Cap Data for Children GICS Codes:')
            #display(return_value)
            # Compute the market weights across all children
            p = return_value.div(return_value.sum(axis=1), axis=0)
            
            return p
        else:
            print('Parent GICS code must be 2, 4, or 6 digits long to have children.')
            return pd.DataFrame()
        
    # Calculate weighted index for a given GICS code and capitalization
    def calculate_weighted_index(self, gics_code, capitalization='Large Cap', period='10y', interval='1d',should_update=False):
        # Retrieve market cap data
        market_cap_data = self.retrieve_market_cap(gics_code, capitalization, period, interval, should_update)
        # Retrieve price data
        price_data = self.retrieve_prices(gics_code, capitalization, price_field='Close', period=period, interval=interval)
        
        
        # Market data is missing dates in price data, we need to align them
        # First, reindex market cap data to match price data dates
        market_cap_data = market_cap_data.reindex(price_data.index).ffill().bfill()
    
        # Calculate total market cap for each date
        total_market_cap = market_cap_data.sum(axis=1)
        # Calculate weights for each company
        weights = market_cap_data.div(total_market_cap, axis=0)
        # Calculate weighted prices
        weighted_prices = price_data.mul(weights)
        # Calculate the weighted index by summing weighted prices across companies
        weighted_index = weighted_prices.sum(axis=1)
        return weighted_index
    
    #takes in a list of gics codes and returns a dataframe with each column as the weighted index for that gics code
    def calculate_weighted_indices(self, gics_codes, capitalization='Large Cap', period='10y', interval='1d',should_update=False):
        indices_dict = {}
        for code in gics_codes:
            index = self.calculate_weighted_index(code, capitalization, period, interval, should_update)
            indices_dict[str(code)] = index
        
        return indices_dict
