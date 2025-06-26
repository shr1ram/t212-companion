"""Statistical calculations for portfolio analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PortfolioStats:
    """Class for calculating portfolio statistics."""
    
    def __init__(self, positions: List[Dict], historical_data: Optional[Dict] = None, 
                 risk_free_rate: float = 0.02):
        """
        Initialize the PortfolioStats class.
        
        Args:
            positions: List of current portfolio positions
            historical_data: Historical order data (optional)
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation (default: 2%)
        """
        self.positions = positions
        self.historical_data = historical_data
        self.risk_free_rate = risk_free_rate
        self.daily_returns = None
        
        # Process the data
        self._process_data()
    
    def _process_data(self):
        """Process the input data and prepare for calculations."""
        # Convert positions to DataFrame for easier manipulation
        self.positions_df = pd.DataFrame(self.positions)
        
        # If historical data is provided, process it
        if self.historical_data and 'items' in self.historical_data:
            self.orders_df = pd.DataFrame(self.historical_data['items'])
            
            # Convert date strings to datetime objects
            date_columns = ['dateCreated', 'dateExecuted', 'dateModified']
            for col in date_columns:
                if col in self.orders_df.columns:
                    self.orders_df[col] = pd.to_datetime(self.orders_df[col])
            
            # Sort by execution date
            if 'dateExecuted' in self.orders_df.columns:
                self.orders_df = self.orders_df.sort_values('dateExecuted')
    
    def calculate_portfolio_value(self) -> float:
        """
        Calculate the current total portfolio value.
        
        Returns:
            Total portfolio value
        """
        if 'quantity' in self.positions_df.columns and 'currentPrice' in self.positions_df.columns:
            return (self.positions_df['quantity'] * self.positions_df['currentPrice']).sum()
        else:
            logger.warning("Required columns not found in positions data")
            return 0.0
    
    def calculate_daily_returns(self, days: int = 365) -> pd.Series:
        """
        Calculate daily returns for the portfolio.
        
        Args:
            days: Number of days to calculate returns for
            
        Returns:
            Series of daily returns
        """
        if self.historical_data is None or 'items' not in self.historical_data:
            logger.warning("Historical data not provided, cannot calculate daily returns")
            return pd.Series()
        
        # Use dateModified if dateExecuted is not available
        date_column = 'dateExecuted' if 'dateExecuted' in self.orders_df.columns and not self.orders_df['dateExecuted'].isna().all() else 'dateModified'
        
        if date_column not in self.orders_df.columns:
            logger.warning(f"Neither dateExecuted nor dateModified found in historical data")
            return pd.Series()
        
        # Use filledValue if fillResult is not available
        value_column = 'fillResult' if 'fillResult' in self.orders_df.columns and not self.orders_df['fillResult'].isna().all() else 'filledValue'
        
        if value_column not in self.orders_df.columns:
            logger.warning(f"Neither fillResult nor filledValue found in historical data")
            return pd.Series()
        
        # Drop rows with missing date or value
        valid_orders = self.orders_df.dropna(subset=[date_column, value_column])
        
        if len(valid_orders) == 0:
            logger.warning("No valid orders with date and value information")
            return pd.Series()
        
        # Group by date and sum the values
        daily_results = valid_orders.groupby(valid_orders[date_column].dt.date)[value_column].sum()
        
        # Calculate daily returns (this is simplified)
        start_date = datetime.now().date() - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=datetime.now().date())
        
        # Create a series with all dates
        all_dates = pd.Series(index=date_range, data=0.0)
        
        # Fill in the dates we have data for
        for date, value in daily_results.items():
            if date in all_dates.index:
                all_dates[date] = value
        
        # Calculate cumulative portfolio value
        cumulative_value = all_dates.cumsum()
        
        # Calculate daily returns based on portfolio value changes
        self.daily_returns = cumulative_value.pct_change().fillna(0)
        
        return self.daily_returns
    
    def calculate_sharpe_ratio(self, days: int = 365) -> float:
        """
        Calculate the Sharpe ratio.
        
        Args:
            days: Number of days to calculate Sharpe ratio for
            
        Returns:
            Sharpe ratio
        """
        if self.daily_returns is None:
            self.calculate_daily_returns(days)
        
        if len(self.daily_returns) == 0:
            logger.warning("No daily returns data available")
            return 0.0
        
        # Calculate annualized return
        annual_return = self.daily_returns.mean() * 252
        
        # Calculate annualized volatility
        annual_volatility = self.daily_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        if annual_volatility == 0:
            logger.warning("Volatility is zero, cannot calculate Sharpe ratio")
            return 0.0
        
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, days: int = 365) -> float:
        """
        Calculate the Sortino ratio (similar to Sharpe but only considers downside risk).
        
        Args:
            days: Number of days to calculate Sortino ratio for
            
        Returns:
            Sortino ratio
        """
        if self.daily_returns is None:
            self.calculate_daily_returns(days)
        
        if len(self.daily_returns) == 0:
            logger.warning("No daily returns data available")
            return 0.0
        
        # Calculate annualized return
        annual_return = self.daily_returns.mean() * 252
        
        # Calculate downside deviation (only negative returns)
        downside_returns = self.daily_returns[self.daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        # Calculate Sortino ratio
        if downside_deviation == 0:
            logger.warning("Downside deviation is zero, cannot calculate Sortino ratio")
            return 0.0
        
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation
        return sortino_ratio
    
    def calculate_drawdowns(self, days: int = 365) -> Tuple[pd.Series, float, float]:
        """
        Calculate drawdowns and maximum drawdown.
        
        Args:
            days: Number of days to calculate drawdowns for
            
        Returns:
            Tuple containing:
            - Series of drawdowns
            - Maximum drawdown value
            - Maximum drawdown duration in days
        """
        if self.daily_returns is None:
            self.calculate_daily_returns(days)
        
        if len(self.daily_returns) == 0:
            logger.warning("No daily returns data available")
            return pd.Series(), 0.0, 0
        
        # Calculate cumulative returns
        cumulative_returns = (1 + self.daily_returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns / running_max) - 1
        
        # Calculate maximum drawdown
        max_drawdown = drawdowns.min()
        
        # Calculate maximum drawdown duration
        is_in_drawdown = drawdowns < 0
        if not is_in_drawdown.any():
            max_duration = 0
        else:
            # Find the longest streak of drawdowns
            drawdown_start = is_in_drawdown.astype(int).diff().fillna(0)
            drawdown_start = drawdown_start[drawdown_start == 1].index
            
            max_duration = 0
            for start in drawdown_start:
                end_idx = is_in_drawdown.loc[start:].idxmin()
                if end_idx is not pd.NaT:
                    duration = (end_idx - start).days
                    max_duration = max(max_duration, duration)
        
        return drawdowns, max_drawdown, max_duration
    
    def calculate_volatility(self, days: int = 365, annualized: bool = True) -> float:
        """
        Calculate portfolio volatility.
        
        Args:
            days: Number of days to calculate volatility for
            annualized: Whether to annualize the volatility
            
        Returns:
            Volatility value
        """
        if self.daily_returns is None:
            self.calculate_daily_returns(days)
        
        if len(self.daily_returns) == 0:
            logger.warning("No daily returns data available")
            return 0.0
        
        volatility = self.daily_returns.std()
        
        if annualized:
            volatility *= np.sqrt(252)  # Annualize using trading days
        
        return volatility
    
    def calculate_beta(self, market_returns: pd.Series) -> float:
        """
        Calculate portfolio beta against a market benchmark.
        
        Args:
            market_returns: Series of market benchmark returns
            
        Returns:
            Beta value
        """
        if self.daily_returns is None:
            self.calculate_daily_returns()
        
        if len(self.daily_returns) == 0:
            logger.warning("No daily returns data available")
            return 0.0
        
        # Align the dates
        aligned_returns = pd.concat([self.daily_returns, market_returns], axis=1).dropna()
        
        if aligned_returns.shape[0] == 0:
            logger.warning("No overlapping dates between portfolio and market returns")
            return 0.0
        
        # Calculate covariance and market variance
        covariance = aligned_returns.iloc[:, 0].cov(aligned_returns.iloc[:, 1])
        market_variance = aligned_returns.iloc[:, 1].var()
        
        if market_variance == 0:
            logger.warning("Market variance is zero, cannot calculate beta")
            return 0.0
        
        beta = covariance / market_variance
        return beta
    
    def calculate_alpha(self, market_returns: pd.Series) -> float:
        """
        Calculate portfolio alpha against a market benchmark.
        
        Args:
            market_returns: Series of market benchmark returns
            
        Returns:
            Alpha value
        """
        if self.daily_returns is None:
            self.calculate_daily_returns()
        
        if len(self.daily_returns) == 0:
            logger.warning("No daily returns data available")
            return 0.0
        
        # Calculate beta
        beta = self.calculate_beta(market_returns)
        
        # Calculate average returns
        portfolio_return = self.daily_returns.mean() * 252  # Annualized
        market_return = market_returns.mean() * 252  # Annualized
        
        # Calculate alpha
        alpha = portfolio_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))
        return alpha
    
    def generate_report(self) -> Dict:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary containing all calculated statistics
        """
        # Calculate daily returns if not already done
        if self.daily_returns is None:
            self.calculate_daily_returns()
        
        # Calculate all statistics
        portfolio_value = self.calculate_portfolio_value()
        sharpe_ratio = self.calculate_sharpe_ratio()
        sortino_ratio = self.calculate_sortino_ratio()
        drawdowns, max_drawdown, max_drawdown_duration = self.calculate_drawdowns()
        volatility = self.calculate_volatility()
        
        # Calculate returns for different periods
        if len(self.daily_returns) > 0:
            daily_return = self.daily_returns.iloc[-1] if len(self.daily_returns) > 0 else 0
            weekly_return = self.daily_returns.iloc[-5:].sum() if len(self.daily_returns) >= 5 else 0
            monthly_return = self.daily_returns.iloc[-21:].sum() if len(self.daily_returns) >= 21 else 0
            yearly_return = self.daily_returns.sum() if len(self.daily_returns) > 0 else 0
        else:
            daily_return = weekly_return = monthly_return = yearly_return = 0
        
        # Compile the report
        report = {
            "portfolio_value": portfolio_value,
            "returns": {
                "daily": daily_return,
                "weekly": weekly_return,
                "monthly": monthly_return,
                "yearly": yearly_return
            },
            "risk_metrics": {
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "max_drawdown_duration": max_drawdown_duration
            }
        }
        
        return report
