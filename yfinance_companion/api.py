"""YFinance API integration for T212 Companion."""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class YFinanceAPI:
    """Client for interacting with the Yahoo Finance API via yfinance."""
    
    # Mapping from T212 tickers to Yahoo Finance tickers
    # The format appears to be: [Symbol][Exchange]_EQ
    # For example, VWRPl_EQ -> VWRP.L (Vanguard FTSE All-World UCITS ETF on London Stock Exchange)
    TICKER_MAPPING = {
        # Common mappings for UK-listed ETFs
        "VWRPl_EQ": "VWRP.L",  # Vanguard FTSE All-World UCITS ETF
        "VWRLl_EQ": "VWRL.L",  # Vanguard FTSE All-World UCITS ETF
        "VUSAl_EQ": "VUSA.L",  # Vanguard S&P 500 UCITS ETF
        "VUKGl_EQ": "VUKG.L",  # Vanguard FTSE 100 UCITS ETF
        "VUAGl_EQ": "VUAG.L",  # Vanguard U.S. Equity Index Fund
        # Add more mappings as needed
    }
    
    def __init__(self):
        """Initialize the YFinance API client."""
        pass
    
    @staticmethod
    def map_ticker(t212_ticker: str) -> str:
        """
        Map Trading 212 ticker to Yahoo Finance ticker format.
        
        Args:
            t212_ticker: Trading 212 ticker format (e.g., "VWRPl_EQ")
            
        Returns:
            Yahoo Finance ticker format (e.g., "VWRP.L")
        """
        if t212_ticker in YFinanceAPI.TICKER_MAPPING:
            return YFinanceAPI.TICKER_MAPPING[t212_ticker]
        
        # If not in mapping, try to derive it
        # Remove _EQ suffix
        base_ticker = t212_ticker.replace("_EQ", "")
        
        # Check if it ends with 'l' (likely London Stock Exchange)
        if base_ticker.endswith('l'):
            return f"{base_ticker[:-1]}.L"
        
        # Default: just return the base ticker without _EQ
        logger.warning(f"No specific mapping for {t212_ticker}, using derived ticker")
        return base_ticker
    
    def get_historical_data(self, t212_ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get historical price data for a Trading 212 ticker.
        
        Args:
            t212_ticker: Trading 212 ticker format
            period: Time period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical price data
        """
        yf_ticker = self.map_ticker(t212_ticker)
        logger.info(f"Getting historical data for {t212_ticker} (Yahoo Finance: {yf_ticker})")
        
        try:
            # Set auto_adjust to True explicitly to avoid warning
            data = yf.download(yf_ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if data.empty:
                logger.warning(f"No data found for {yf_ticker}")
                return pd.DataFrame()
            
            # Handle the case where columns are tuples (metric, ticker)
            if isinstance(data.columns[0], tuple):
                # Rename columns to just the metric name for easier access
                data.columns = [col[0] for col in data.columns]
            
            # Add the original ticker for reference
            data['t212_ticker'] = t212_ticker
            data['yf_ticker'] = yf_ticker
            
            return data
        except Exception as e:
            logger.error(f"Error downloading data for {yf_ticker}: {e}")
            return pd.DataFrame()
    
    def get_portfolio_historical_data(self, positions: List[Dict], period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Get historical price data for all positions in a portfolio.
        
        Args:
            positions: List of position dictionaries from T212 API
            period: Time period to download
            interval: Data interval
            
        Returns:
            Dictionary mapping T212 tickers to DataFrames with historical data
        """
        historical_data = {}
        
        for position in positions:
            ticker = position.get('ticker')
            if not ticker:
                logger.warning(f"Position missing ticker: {position}")
                continue
                
            data = self.get_historical_data(ticker, period=period, interval=interval)
            if not data.empty:
                historical_data[ticker] = data
        
        return historical_data


# The PortfolioAnalyzer class has been moved to the portfolio_analysis module
