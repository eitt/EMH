import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_data(tickers, start_date, end_date, output_dir='data/raw'):
    """
    Downloads historical market data for specified tickers.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_file = os.path.join(output_dir, f'market_data_{timestamp}.csv')
    
    logger.info(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    
    try:
        # Download Close prices and Volume
        df = yf.download(tickers, start=start_date, end=end_date)
        
        if df.empty:
            logger.error("No data downloaded. Check tickers and date range.")
            return None
            
        # Standardize format: Multi-index columns [PriceType, Ticker]
        # We ensure it's saved with a clear index
        df.to_csv(data_file)
        logger.info(f"Raw data saved to {data_file}")
        
        # Also save a symlink or 'latest' version for easier pipeline access
        latest_file = os.path.join(output_dir, 'latest_raw.csv')
        df.to_csv(latest_file)
        
        return data_file
    except Exception as e:
        logger.error(f"Error during data ingestion: {str(e)}")
        return None

if __name__ == "__main__":
    # Default scope for LatAm
    DEFAULT_TICKERS = ['EWZ', 'EWW', 'ECH', 'GXG']
    START = '2015-01-01'
    END = '2024-01-01'
    
    download_data(DEFAULT_TICKERS, START, END)
