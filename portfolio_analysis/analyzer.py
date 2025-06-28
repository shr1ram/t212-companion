"""Portfolio analyzer module for T212 Companion."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Import YFinanceAPI from yfinance_companion
from yfinance_companion.api import YFinanceAPI


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
        
    def calculate_mean_annual_return(self, returns: pd.Series = None) -> float:
        """
        Calculate mean annual return from daily returns.
        
        Args:
            returns: Series of returns (if None, uses portfolio returns)
            
        Returns:
            Mean annual return as a percentage
        """
        if returns is None:
            if self.portfolio_returns is None:
                self.calculate_portfolio_returns()
            returns = self.portfolio_returns
            
        if returns is None or returns.empty:
            logger.warning("No returns data available to calculate mean annual return")
            return 0.0
            
        try:
            # Calculate mean daily return
            mean_daily_return = returns.mean()
            
            # Convert to annual return (assuming 256 trading days)
            annual_return = mean_daily_return * 256
            
            # Convert to percentage
            return annual_return * 100
        except Exception as e:
            logger.error(f"Error calculating mean annual return: {e}")
            return 0.0
            
    def calculate_standard_deviation(self, returns: pd.Series = None, annualized: bool = True) -> float:
        """
        Calculate standard deviation of returns.
        
        Args:
            returns: Series of returns (if None, uses portfolio returns)
            annualized: Whether to annualize the standard deviation
            
        Returns:
            Standard deviation as a percentage
        """
        if returns is None:
            if self.portfolio_returns is None:
                self.calculate_portfolio_returns()
            returns = self.portfolio_returns
            
        if returns is None or returns.empty:
            logger.warning("No returns data available to calculate standard deviation")
            return 0.0
            
        try:
            # Calculate standard deviation
            std_dev = returns.std()
            
            # Annualize if requested (assuming 256 trading days)
            if annualized:
                std_dev = std_dev * np.sqrt(256)
                
            # Convert to percentage
            return std_dev * 100
        except Exception as e:
            logger.error(f"Error calculating standard deviation: {e}")
            return 0.0
            
    def calculate_drawdowns(self, returns: pd.Series = None) -> pd.Series:
        """
        Calculate drawdowns from a series of returns.
        
        Args:
            returns: Series of returns (if None, uses portfolio returns)
            
        Returns:
            Series of drawdowns
        """
        if returns is None:
            if self.portfolio_returns is None:
                self.calculate_portfolio_returns()
            returns = self.portfolio_returns
            
        if returns is None or returns.empty:
            logger.warning("No returns data available to calculate drawdowns")
            return pd.Series()
            
        try:
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.cummax()
            
            # Calculate drawdowns
            drawdowns = (cum_returns - running_max) / running_max
            
            return drawdowns
        except Exception as e:
            logger.error(f"Error calculating drawdowns: {e}")
            return pd.Series()
            
    def calculate_average_drawdown(self, returns: pd.Series = None) -> float:
        """
        Calculate average drawdown from a series of returns.
        
        Args:
            returns: Series of returns (if None, uses portfolio returns)
            
        Returns:
            Average drawdown as a percentage
        """
        drawdowns = self.calculate_drawdowns(returns)
        
        if drawdowns.empty:
            return 0.0
            
        # Filter for only negative values (actual drawdowns)
        negative_drawdowns = drawdowns[drawdowns < 0]
        
        if negative_drawdowns.empty:
            return 0.0
            
        # Calculate average and convert to percentage
        return negative_drawdowns.mean() * 100
        
    def calculate_max_drawdown(self, returns: pd.Series = None) -> float:
        """
        Calculate maximum drawdown from a series of returns.
        
        Args:
            returns: Series of returns (if None, uses portfolio returns)
            
        Returns:
            Maximum drawdown as a percentage
        """
        drawdowns = self.calculate_drawdowns(returns)
        
        if drawdowns.empty:
            return 0.0
            
        # Find minimum (worst) drawdown and convert to percentage
        return drawdowns.min() * 100
        
    def calculate_skew(self, returns: pd.Series = None) -> float:
        """
        Calculate skewness of returns distribution.
        
        Args:
            returns: Series of returns (if None, uses portfolio returns)
            
        Returns:
            Skewness value
        """
        if returns is None:
            if self.portfolio_returns is None:
                self.calculate_portfolio_returns()
            returns = self.portfolio_returns
            
        if returns is None or returns.empty:
            logger.warning("No returns data available to calculate skew")
            return 0.0
            
        try:
            return returns.skew()
        except Exception as e:
            logger.error(f"Error calculating skew: {e}")
            return 0.0
            
    def calculate_tail_metrics(self, returns: pd.Series = None, lower_percentile: float = 5.0, upper_percentile: float = 95.0) -> Dict[str, float]:
        """
        Calculate relative lower and upper fat tail ratios (See R Carver AFTS).
        
        Args:
            returns: Series of returns (if None, uses portfolio returns)
            lower_percentile: Lower fat tail ratio
            upper_percentile: Upper fat tail ratio
            
        Returns:
            Dictionary with lower and upper tail ratios
        """
        if returns is None:
            if self.portfolio_returns is None:
                self.calculate_portfolio_returns()
            returns = self.portfolio_returns
            
        if returns is None or returns.empty:
            logger.warning("No returns data available to calculate tail metrics")
            return {"lower_tail": 0.0, "upper_tail": 0.0}
            
        try:
            percentile_1 = returns.quantile(1 / 100) * 100
            percentile_30 = returns.quantile(30 / 100) * 100
            percentile_70 = returns.quantile(70 / 100) * 100
            percentile_99 = returns.quantile(99 / 100) * 100
            
            lower_percentile_ratio = (percentile_1 / percentile_30)
            upper_percentile_ratio = (percentile_99 / percentile_70)

            gaussian_ratio = 4.43

            relative_lower_fat_tail_ratio = lower_percentile_ratio / gaussian_ratio
            relative_upper_fat_tail_ratio = upper_percentile_ratio / gaussian_ratio

            lower_tail = relative_lower_fat_tail_ratio
            upper_tail = relative_upper_fat_tail_ratio

            return {"lower_tail": lower_tail, "upper_tail": upper_tail}
        except Exception as e:
            logger.error(f"Error calculating tail metrics: {e}")
            return {"lower_tail": 0.0, "upper_tail": 0.0}
            
    # Turnover calculation removed as requested
        
    def calculate_all_metrics(self, returns: pd.Series = None) -> Dict[str, float]:
        """
        Calculate all portfolio metrics in one call.
        
        Args:
            returns: Series of returns (if None, uses portfolio returns)
            
        Returns:
            Dictionary of all calculated metrics
        """
        if returns is None:
            if self.portfolio_returns is None:
                self.calculate_portfolio_returns()
            returns = self.portfolio_returns
            
        if returns is None or returns.empty:
            logger.warning("No returns data available to calculate metrics")
            return {}
            
        # Calculate all metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        mean_annual_return = self.calculate_mean_annual_return(returns)
        std_dev = self.calculate_standard_deviation(returns)
        avg_drawdown = self.calculate_average_drawdown(returns)
        max_drawdown = self.calculate_max_drawdown(returns)
        skew = self.calculate_skew(returns)
        tail_metrics = self.calculate_tail_metrics(returns)
        
        # Combine all metrics into a dictionary
        metrics = {
            "sharpe_ratio": sharpe_ratio,
            "mean_annual_return": mean_annual_return,
            "standard_deviation": std_dev,
            "average_drawdown": avg_drawdown,
            "max_drawdown": max_drawdown,
            "skew": skew,
            "lower_tail": tail_metrics["lower_tail"],
            "upper_tail": tail_metrics["upper_tail"]
        }
        
        return metrics
