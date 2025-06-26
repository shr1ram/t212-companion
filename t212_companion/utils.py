"""Utility functions for the Trading 212 Companion."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import os
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_to_csv(data: Union[List[Dict], pd.DataFrame], filename: str, directory: str = "data") -> str:
    """
    Save data to a CSV file.
    
    Args:
        data: Data to save (list of dictionaries or DataFrame)
        filename: Name of the file (without extension)
        directory: Directory to save the file in
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Convert to DataFrame if necessary
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(directory, f"{filename}_{timestamp}.csv")
    
    # Save to CSV
    data.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}")
    
    return filepath


def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = pd.read_csv(filepath)
    logger.info(f"Data loaded from {filepath}")
    
    return data


def save_to_json(data: Union[Dict, List[Dict]], filename: str, directory: str = "data") -> str:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filename: Name of the file (without extension)
        directory: Directory to save the file in
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(directory, f"{filename}_{timestamp}.json")
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"Data saved to {filepath}")
    
    return filepath


def load_from_json(filepath: str) -> Union[Dict, List]:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Data from the JSON file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Data loaded from {filepath}")
    
    return data


def calculate_cagr(initial_value: float, final_value: float, years: float) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        initial_value: Initial investment value
        final_value: Final investment value
        years: Number of years
        
    Returns:
        CAGR as a decimal
    """
    if initial_value <= 0 or years <= 0:
        logger.warning("Invalid input for CAGR calculation")
        return 0.0
    
    cagr = (final_value / initial_value) ** (1 / years) - 1
    return cagr


def calculate_annualized_return(returns: pd.Series) -> float:
    """
    Calculate annualized return from a series of returns.
    
    Args:
        returns: Series of returns (can be daily, monthly, etc.)
        
    Returns:
        Annualized return as a decimal
    """
    if len(returns) == 0:
        logger.warning("Empty returns series")
        return 0.0
    
    # Determine the frequency of returns
    if isinstance(returns.index, pd.DatetimeIndex):
        # Calculate average days between observations
        if len(returns) > 1:
            avg_days = (returns.index[-1] - returns.index[0]).days / (len(returns) - 1)
        else:
            avg_days = 1
        
        # Determine annualization factor
        if avg_days < 3:  # Daily
            factor = 252
        elif avg_days < 10:  # Weekly
            factor = 52
        elif avg_days < 45:  # Monthly
            factor = 12
        else:  # Quarterly or less frequent
            factor = 4
    else:
        # Default to daily
        factor = 252
    
    # Calculate annualized return
    cumulative_return = (1 + returns).prod() - 1
    periods = len(returns)
    
    annualized_return = (1 + cumulative_return) ** (factor / periods) - 1
    return annualized_return


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate the win rate (percentage of positive returns).
    
    Args:
        returns: Series of returns
        
    Returns:
        Win rate as a decimal
    """
    if len(returns) == 0:
        logger.warning("Empty returns series")
        return 0.0
    
    win_count = (returns > 0).sum()
    win_rate = win_count / len(returns)
    
    return win_rate


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Calculate the profit factor (gross profits / gross losses).
    
    Args:
        returns: Series of returns
        
    Returns:
        Profit factor
    """
    if len(returns) == 0:
        logger.warning("Empty returns series")
        return 0.0
    
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return float('inf')  # No losses
    
    profit_factor = profits / losses
    return profit_factor


def calculate_calmar_ratio(returns: pd.Series, max_drawdown: float) -> float:
    """
    Calculate the Calmar ratio (annualized return / maximum drawdown).
    
    Args:
        returns: Series of returns
        max_drawdown: Maximum drawdown as a positive decimal
        
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        logger.warning("Maximum drawdown is zero")
        return float('inf')
    
    annualized_return = calculate_annualized_return(returns)
    calmar_ratio = annualized_return / abs(max_drawdown)
    
    return calmar_ratio


def resample_returns(returns: pd.Series, freq: str = 'M') -> pd.Series:
    """
    Resample returns to a different frequency.
    
    Args:
        returns: Series of returns
        freq: Frequency to resample to ('D' for daily, 'W' for weekly, 'M' for monthly, etc.)
        
    Returns:
        Resampled returns
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        logger.warning("Returns index is not a DatetimeIndex")
        return returns
    
    # Resample by compounding returns
    resampled = returns.resample(freq).apply(lambda x: (1 + x).prod() - 1)
    
    return resampled
