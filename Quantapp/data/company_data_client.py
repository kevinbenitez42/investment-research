import json
import os

import pandas as pd
import yfinance as yf
try:
    from quickfs import QuickFS
except ImportError:
    QuickFS = None

class CompanyDataClient:
   
    def __init__(self, ticker_str, client=None,save_path=None,debug=False):
        """
        Initializes the CompanyDataClient class with a ticker and an optional QuickFS client.
        
        Parameters:
            ticker (str): The stock ticker symbol for the company.
            client (QuickFSClient, optional): An instance of QuickFSClient to interact with the QuickFS API.
        """
        self.ticker_str = ticker_str
        self.debug = debug
        if client is not None:
            self.client = client
        else:
            if QuickFS is None:
                raise ImportError("quickfs is required when no client is provided.")
            self.client = QuickFS()
            self._log("No QuickFS client provided, initializing default client.")

        self.save_path = save_path if save_path is not None else os.getcwd()

    def _log(self, message):
        if self.debug:
            print(message)
        
    def get_metrics(self):
        """
        Retrieves the metrics available from the QuickFS API.
        
        Parameters:
            client (QuickFSClient): The client object used to interact with the API.
        
        Returns:
            list: A list of dictionaries containing the available metrics and their descriptions.
        """
        # Retrieve the metrics available from the API
        metrics = self.client.get_available_metrics()
        
        # Return the list of metrics
        return metrics
    
    def get_latest_earnings_date(self):
        """
        Retrieves the most recent earnings date for the company.
        
        Returns:
            str: The most recent earnings date in 'YYYY-MM-DD' format or None if no dates found.
        """
        earnings_dates = yf.Ticker(self.ticker_str).earnings_dates.dropna().index
        if len(earnings_dates) > 0:
            last_earnings_date = str(earnings_dates[0].strftime('%Y-%m-%d'))
            return last_earnings_date
        return None
    
    def process_full_data(self, full_data, client):
        """
        Processes the full_data dictionary to extract company metadata and financials.
        It converts financial data (annual and quarterly) into DataFrames with sorted columns,
        and then separates them into dictionaries based on statement types using available metrics.
        
        Parameters:
            full_data (dict): Dictionary containing keys 'metadata' and 'financials' (with keys 'annual' and 'quarterly').
            client: Parameter used to retrieve metrics via get_metrics(client).
        
        Returns:
            dict: A dictionary containing:
                'ticker': company ticker,
                'metadata': company metadata,
                'financials_annual': dictionary of annual financial DataFrames,
                'financials_quarterly': dictionary of quarterly financial DataFrames.
        """
        def create_financials_dict(financials_df, client):
            # Retrieve the metrics from the client using get_metrics
            
            metrics = self.get_metrics()
            # Identify metrics that are not present in the DataFrame
            metrics_not_in_df = []
            for field in metrics:
                if field['metric'] not in financials_df.columns:
                    metrics_not_in_df.append(field['metric'])

            # Filter out metrics not in the DataFrame and sort the remaining metrics
            metrics = [field for field in metrics if field['metric'] not in metrics_not_in_df]
            metrics = sorted(metrics, key=lambda x: x['metric'])
            

            
            
            # Extract metrics based on statement type
            metrics_income_statement    = [field['metric'] for field in metrics if field['statement_type'] == 'income_statement']
            metrics_balance_sheet       = [field['metric'] for field in metrics if field['statement_type'] == 'balance_sheet']
            metrics_cash_flow_statement = [field['metric'] for field in metrics if field['statement_type'] == 'cash_flow_statement']
            metrics_computed             = [field['metric'] for field in metrics if field['statement_type'] == 'computed']
            metrics_misc                = [field['metric'] for field in metrics if field['statement_type'] == 'misc']
    
            # Create separate DataFrames for each statement type by concatenating metrics_misc with the respective group
            income_statement_df     = pd.concat([financials_df[metrics_misc], financials_df[metrics_income_statement]], axis=1)
            balance_sheet_df        = pd.concat([financials_df[metrics_misc], financials_df[metrics_balance_sheet]], axis=1)
            cash_flow_statement_df  = pd.concat([financials_df[metrics_misc], financials_df[metrics_cash_flow_statement]], axis=1)
            computed_df              = pd.concat([financials_df[metrics_misc], financials_df[metrics_computed]], axis=1)
            misc_df                 = financials_df[metrics_misc]
            
            #fiscal_year_key
            #fiscal_year_number
            #fiscal_quarter_key
            #fiscal_quarter_number 
            
            #check if fiscal_year_key, fiscal_year_number, fiscal_quarter_key, fiscal_quarter_number are in the financials_df
            #if they are, add columns to each dataframe
            
            if 'fiscal_year_key' in financials_df.columns:
                income_statement_df['fiscal_year_key'] = financials_df['fiscal_year_key']
                balance_sheet_df['fiscal_year_key'] = financials_df['fiscal_year_key']
                cash_flow_statement_df['fiscal_year_key'] = financials_df['fiscal_year_key']
                computed_df['fiscal_year_key'] = financials_df['fiscal_year_key']
                misc_df['fiscal_year_key'] = financials_df['fiscal_year_key']
                
            if 'fiscal_year_number' in financials_df.columns:
                income_statement_df['fiscal_year_number'] = financials_df['fiscal_year_number']
                balance_sheet_df['fiscal_year_number'] = financials_df['fiscal_year_number']
                cash_flow_statement_df['fiscal_year_number'] = financials_df['fiscal_year_number']
                computed_df['fiscal_year_number'] = financials_df['fiscal_year_number']
                misc_df['fiscal_year_number'] = financials_df['fiscal_year_number']
                
            if 'fiscal_quarter_key' in financials_df.columns:
                income_statement_df['fiscal_quarter_key'] = financials_df['fiscal_quarter_key']
                balance_sheet_df['fiscal_quarter_key'] = financials_df['fiscal_quarter_key']
                cash_flow_statement_df['fiscal_quarter_key'] = financials_df['fiscal_quarter_key']
                computed_df['fiscal_quarter_key'] = financials_df['fiscal_quarter_key']
                misc_df['fiscal_quarter_key'] = financials_df['fiscal_quarter_key']
                
            if 'fiscal_quarter_number' in financials_df.columns:
                income_statement_df['fiscal_quarter_number'] = financials_df['fiscal_quarter_number']
                balance_sheet_df['fiscal_quarter_number'] = financials_df['fiscal_quarter_number']
                cash_flow_statement_df['fiscal_quarter_number'] = financials_df['fiscal_quarter_number']
                computed_df['fiscal_quarter_number'] = financials_df['fiscal_quarter_number']
                misc_df['fiscal_quarter_number'] = financials_df['fiscal_quarter_number']
                
            # Return a dictionary with the separated DataFrames
            return {
                'income_statement': income_statement_df,
                'balance_sheet': balance_sheet_df,
                'cash_flow_statement': cash_flow_statement_df,
                'computed': computed_df,
                'misc': misc_df
            }
       
        # Get ticker; default to 'AAPL' if not provided in full_data
        ticker = full_data['metadata']['symbol']
        
        # Extract metadata and financials from full_data
        company_metadata     = full_data['metadata']
        financials_annual    = full_data['financials']['annual']
        financials_quarterly = full_data['financials']['quarterly']
        
        
     
        if "preliminary" in financials_annual:
            del financials_annual["preliminary"] 
        if "preliminary" in financials_quarterly:
            del financials_quarterly["preliminary"]
        #display(financials_annual)
        #display(financials_quarterly)
        # Create DataFrames for annual and quarterly financials with sorted columns
        financials_annual_df = pd.DataFrame(financials_annual).reindex(
            sorted(pd.DataFrame(financials_annual).columns), axis=1
        )
        financials_quarterly_df = pd.DataFrame(financials_quarterly).reindex(
            sorted(pd.DataFrame(financials_quarterly).columns), axis=1
        )
        

        
        # Create dictionaries for annual and quarterly financials
        financials_annual_dict    = create_financials_dict(financials_annual_df, self.client)
        financials_quarterly_dict = create_financials_dict(financials_quarterly_df, self.client)
        
        return {
            'ticker': ticker,
            'metadata': company_metadata,
            'financials_annual': financials_annual_dict,
            'financials_quarterly': financials_quarterly_dict
        }
    
    #do not call this function directly, use retrieve_data instead        
    def retrieve_data_from_API(self):
        if self.debug:
            self._log('Querying data API!')
            
        #convert dictionary to 
        data = self.client.get_data_full(symbol=self.ticker_str)
        
        if data is None:
            if self.debug:
                self._log('No data returned from API!')
            return None
        
        data = self.process_full_data(data, self.client)
        
        if self.debug:
            self._log('Data retreived successfully!\n')
        self.save_company_info(self.ticker_str, data['financials_annual'], data['financials_quarterly'], data['metadata'])
        if self.debug:
            self._log('Data saved successfully!\n')
        return data
    
    def retrieve_data(self, data_type='annual', statement_type='income_statement', should_update=False):

        ticker_folder = os.path.join(self.save_path, 'company_data', self.ticker_str)
        
        # Check if the ticker folder exists; if not, create it
        if self.debug:
            self._log(f'Checking for company folder at: {ticker_folder}\n')
        if not os.path.exists(ticker_folder):
            os.makedirs(ticker_folder)
            if self.debug:
                self._log(f'Company folder does not exist, creating it now! for {self.ticker_str}\n')
        else:
            if self.debug:
                self._log('Company folder exists\n')

        # Check if the company folder is empty
        if len(os.listdir(ticker_folder)) == 0:
            if self.debug:
                self._log(ticker_folder)
                self._log('No data found for this company\n')
            self.retrieve_data_from_API()
        else:
            if self.debug:
                self._log('Data found for this company\n')
            
        if should_update:
            if self.debug:
                self._log('Updating data from API as per user request\n')
            self.retrieve_data_from_API()
        else:
            if self.debug:
                self._log('Set should_update to True to manually update data\n')
            
            
        #if no data found in the folder even after retrieving from API, return None
        if len(os.listdir(ticker_folder)) == 0:
            if self.debug:
                self._log('No data found for this company even after retrieving from API\n')
            return None
        
        # Check if the data is up to date; if not, retrieve it from the API
        # (This comment is a reminder for future implementation if needed.)

        # If the data exists and is up to date, load it from disk according to the fiscal period
        if data_type == 'annual':
            if self.debug:
                self._log('Retrieving annual data\n')
            return pd.read_csv(os.path.join(ticker_folder, data_type, statement_type + '.csv'))
        elif data_type == 'quarterly':
            if self.debug:
                self._log('Retrieving quarterly data\n')
            return pd.read_csv(os.path.join(ticker_folder, data_type, statement_type + '.csv'))
        elif data_type == 'metadata':
            if self.debug:
                self._log('Retrieving metadata\n')
            with open(os.path.join(ticker_folder, 'metadata.json')) as f:
                return json.load(f)
        else:
            if self.debug:
                self._log('Invalid fiscal period\n')

    def save_company_info(self,ticker, financials_data_annual, financials_data_quarterly, company_metadata):
        """
        Saves company financial data (annual and quarterly) and metadata to disk.
        
        This function:
        - Creates a folder for the given ticker under 'company_data' if it doesn't exist.
        - For each period ('annual' and 'quarterly'), creates a subfolder and saves each
            financial DataFrame as a CSV file.
        - Saves the company metadata as a JSON file in the ticker folder.
        
        Parameters:
        ticker (str): The company ticker.
        financials_data_annual (dict): Dictionary of annual financial DataFrames.
        financials_data_quarterly (dict): Dictionary of quarterly financial DataFrames.
        company_metadata (dict): Dictionary containing company metadata.
        """
        save_path = self.save_path
        parent_folder = os.path.join(save_path, 'company_data')
        
        #parent_folder = 'company_data'
        ticker_folder = os.path.join(parent_folder, ticker)
        if not os.path.exists(ticker_folder):
            os.makedirs(ticker_folder)
            self._log(f"Created folder: {ticker_folder}")

        # Function to save financial data for a given period
        def save_financials_for_period(financials_data, period):
            period_folder = os.path.join(ticker_folder, period)
            if not os.path.exists(period_folder):
                os.makedirs(period_folder)
                self._log(f"Created folder for {period} data: {period_folder}")

            for key, df in financials_data.items():
                file_path = os.path.join(period_folder, f'{key}.csv')
                df.to_csv(file_path)
                self._log(f"{key} data saved to {file_path}")

        # Save annual financials
        save_financials_for_period(financials_data_annual, 'annual')

        # Save quarterly financials
        save_financials_for_period(financials_data_quarterly, 'quarterly')

        # Save company metadata to JSON
        metadata_path = os.path.join(ticker_folder, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(company_metadata, f, indent=4)
        self._log(f"Metadata saved to {metadata_path}")
    
    def retrieve_market_cap(self,data_type='quarterly',should_update=False):
        misc_data = self.retrieve_data(data_type=data_type, statement_type='misc', should_update=should_update)
        #compute market cap as shares_eop * period_end_price
        if misc_data is None:
            self._log(f'No misc data found for {self.ticker_str}')
            return None
        misc_data['market_cap'] = misc_data['shares_eop'] * misc_data['period_end_price']
        #set period_end_date as datetime
        misc_data['period_end_date'] = pd.to_datetime(misc_data['period_end_date'])
        return misc_data[['period_end_date', 'shares_eop','period_end_price', 'market_cap']]   
