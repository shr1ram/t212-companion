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
    
    # Create filepath without timestamp
    filepath = os.path.join(directory, f"{filename}.csv")
    
    # Save to CSV
    data.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}")
    
    # Save last update timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(directory, "last_update.txt"), 'w') as f:
        f.write(f"Last updated: {timestamp}")
    
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
    
    # Create filepath without timestamp
    filepath = os.path.join(directory, f"{filename}.json")
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"Data saved to {filepath}")
    
    # Save last update timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(directory, "last_update.txt"), 'w') as f:
        f.write(f"Last updated: {timestamp}")
    
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