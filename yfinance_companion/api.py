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


class PortfolioAnalyzer:
    """Class for analyzing portfolio performance using Yahoo Finance data."""
    
    def __init__(self, positions: List[Dict], risk_free_rate: float = 0.02):
        """
        Initialize the PortfolioAnalyzer class.
        
        Args:
            positions: List of position dictionaries from T212 API
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation (default: 2%)
        """
        self.positions = positions
        self.risk_free_rate = risk_free_rate
        self.yf_api = YFinanceAPI()
        self.historical_data = None
        self.portfolio_returns = None
        
    def load_historical_data(self, period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Load historical price data for all positions in the portfolio.
        
        Args:
            period: Time period to download
            interval: Data interval
            
        Returns:
            Dictionary mapping tickers to historical data
        """
        self.historical_data = self.yf_api.get_portfolio_historical_data(
            self.positions, period=period, interval=interval
        )
        return self.historical_data
    
    def calculate_returns(self) -> Dict[str, pd.Series]:
        """
        Calculate daily returns for each position and the portfolio.
        
        Returns:
            Dictionary mapping tickers to return series
        """
        if not self.historical_data:
            logger.warning("Historical data not loaded. Call load_historical_data() first.")
            return {}
        
        returns = {}
        
        # Calculate returns for each position
        for ticker, data in self.historical_data.items():
            if not data.empty:
                # Use Close price if Adj Close is not available
                price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                
                if price_col in data.columns:
                    # Make sure we have enough data points
                    if len(data) > 5:  # Require at least 5 data points
                        return_series = data[price_col].pct_change().dropna()
                        if not return_series.empty:
                            returns[ticker] = return_series
                    else:
                        logger.warning(f"Not enough data points for {ticker} to calculate returns")
                else:
                    logger.warning(f"No price column found for {ticker}")
        
        if not returns:
            logger.warning("No valid return series could be calculated for any instrument")
            
        return returns
    
    def calculate_portfolio_returns(self) -> pd.Series:
        """
        Calculate weighted portfolio returns based on position quantities.
        
        Returns:
            Series of portfolio returns
        """
        if not self.historical_data:
            logger.warning("Historical data not loaded. Call load_historical_data() first.")
            return pd.Series()
        
        # Create a dictionary to map tickers to their quantities
        ticker_quantities = {position['ticker']: position['quantity'] for position in self.positions}
        
        # Get returns for each position
        returns = self.calculate_returns()
        
        if not returns:
            logger.warning("No valid returns data available for any position")
            self.portfolio_returns = pd.Series()
            return self.portfolio_returns
        
        # Create a DataFrame of all returns
        returns_df = pd.DataFrame({ticker: series for ticker, series in returns.items()})
        
        if returns_df.empty:
            logger.warning("Empty returns DataFrame")
            self.portfolio_returns = pd.Series()
            return self.portfolio_returns
            
        # Calculate portfolio weights
        total_quantity = sum(ticker_quantities.values())
        if total_quantity == 0:
            logger.warning("Total position quantity is zero, cannot calculate weights")
            self.portfolio_returns = pd.Series()
            return self.portfolio_returns
            
        weights = {ticker: ticker_quantities.get(ticker, 0) / total_quantity 
                  for ticker in returns_df.columns}
        
        # Calculate weighted returns
        for ticker in returns_df.columns:
            returns_df[ticker] = returns_df[ticker] * weights.get(ticker, 0)
        
        # Sum across all positions to get portfolio returns
        self.portfolio_returns = returns_df.sum(axis=1)
        
        if self.portfolio_returns.empty:
            logger.warning("Calculated portfolio returns series is empty")
            
        return self.portfolio_returns
    
    def calculate_sharpe_ratio(self, returns: pd.Series = None) -> float:
        """
        Calculate Sharpe ratio for a return series.
        
        Args:
            returns: Series of returns (if None, uses portfolio returns)
            
        Returns:
            Sharpe ratio
        """
        if returns is None:
            if self.portfolio_returns is None:
                self.calculate_portfolio_returns()
            returns = self.portfolio_returns
        
        if returns is None or returns.empty:
            logger.warning("Empty returns series, cannot calculate Sharpe ratio")
            return 0.0
        
        # Calculate excess returns over risk-free rate
        # Convert annual risk-free rate to daily
        daily_risk_free = (1 + self.risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_risk_free
        
        # Avoid division by zero
        std_dev = excess_returns.std()
        if std_dev == 0 or np.isnan(std_dev):
            logger.warning("Standard deviation is zero or NaN, cannot calculate Sharpe ratio")
            return 0.0
            
        # Calculate Sharpe ratio (annualized)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / std_dev
        
        # Handle NaN result
        if np.isnan(sharpe_ratio):
            logger.warning("Sharpe ratio calculation resulted in NaN")
            return 0.0
            
        return sharpe_ratio
    
    def calculate_all_sharpe_ratios(self) -> Dict[str, float]:
        """
        Calculate Sharpe ratios for all positions and the portfolio.
        
        Returns:
            Dictionary mapping tickers to Sharpe ratios
        """
        returns = self.calculate_returns()
        
        sharpe_ratios = {}
        for ticker, series in returns.items():
            sharpe_ratios[ticker] = self.calculate_sharpe_ratio(series)
        
        # Add portfolio Sharpe ratio
        if self.portfolio_returns is None:
            self.calculate_portfolio_returns()
        
        sharpe_ratios['Portfolio'] = self.calculate_sharpe_ratio(self.portfolio_returns)
        
        return sharpe_ratios
    
    def plot_prices(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot historical prices for all positions.
        
        Args:
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        if not self.historical_data:
            logger.warning("Historical data not loaded. Call load_historical_data() first.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        has_data = False
        for ticker, data in self.historical_data.items():
            if not data.empty:
                # Use Close price if Adj Close is not available
                price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                
                if price_col in data.columns:
                    # Normalize to starting price = 100 for better comparison
                    normalized_price = data[price_col] / data[price_col].iloc[0] * 100
                    ax.plot(normalized_price.index, normalized_price, label=ticker)
                    has_data = True
        
        if not has_data:
            logger.warning("No price data available to plot")
            # Add a dummy text to the plot
            ax.text(0.5, 0.5, 'No price data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            
        ax.set_title('Normalized Price Performance (Starting Value = 100)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price')
        
        # Only add legend if we have data
        if has_data:
            ax.legend()
            
        ax.grid(True)
        
        return fig
    
    def plot_portfolio_performance(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot portfolio cumulative returns.
        
        Args:
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        if self.portfolio_returns is None:
            self.calculate_portfolio_returns()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if self.portfolio_returns is None or self.portfolio_returns.empty:
            logger.warning("Empty portfolio returns, cannot plot performance")
            # Add a dummy text to the plot
            ax.text(0.5, 0.5, 'No portfolio return data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
        else:
            # Calculate cumulative returns
            cumulative_returns = (1 + self.portfolio_returns).cumprod()
            
            ax.plot(cumulative_returns.index, cumulative_returns, label='Portfolio')
            ax.legend()
        
        ax.set_title('Portfolio Cumulative Returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.grid(True)
        
        return fig
    
    def plot_sharpe_ratios(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot Sharpe ratios for all positions and the portfolio.
        
        Args:
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        sharpe_ratios = self.calculate_all_sharpe_ratios()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if not sharpe_ratios or all(v == 0 for v in sharpe_ratios.values()):
            logger.warning("No valid Sharpe ratios to plot")
            # Add a dummy text to the plot
            ax.text(0.5, 0.5, 'No valid Sharpe ratios available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
        else:
            # Sort by Sharpe ratio
            sorted_sharpe = {k: v for k, v in sorted(sharpe_ratios.items(), key=lambda item: item[1])}
            
            # Create bar chart
            bars = ax.bar(sorted_sharpe.keys(), sorted_sharpe.values())
            
            # Highlight portfolio bar
            if 'Portfolio' in sorted_sharpe:
                portfolio_idx = list(sorted_sharpe.keys()).index('Portfolio')
                bars[portfolio_idx].set_color('red')
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
                            
            plt.xticks(rotation=45)
        
        ax.set_title('Sharpe Ratios')
        ax.set_xlabel('Ticker')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, axis='y')
        
        return fig
