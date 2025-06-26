"""Trading 212 API client module."""

import os
import json
import logging
from typing import Dict, List, Optional, Union
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class T212API:
    """Client for interacting with the Trading 212 API."""

    BASE_URL = "https://live.trading212.com/api/v0"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Trading 212 API client.
        
        Args:
            api_key: Trading 212 API key. If not provided, will look for T212_API_KEY in environment variables.
        """
        self.api_key = api_key or os.getenv("T212_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and T212_API_KEY not found in environment variables")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"{self.api_key}",
            "Content-Type": "application/json"
        })
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                     data: Optional[Dict] = None) -> Dict:
        """
        Make a request to the Trading 212 API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            
        Returns:
            API response as a dictionary
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params)
            elif method.upper() == "POST":
                response = self.session.post(url, params=params, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response content: {e.response.text}")
            raise
    
    def get_account_data(self) -> Dict:
        """
        Get account metadata and cash information.
        
        Returns:
            Dictionary containing account metadata and cash information
        """
        metadata = self._make_request("GET", "equity/account/info")
        cash = self._make_request("GET", "equity/account/cash")
        
        return {
            "metadata": metadata,
            "cash": cash
        }
    
    def get_positions(self) -> List[Dict]:
        """
        Get all open positions in the portfolio.
        
        Returns:
            List of positions
        """
        return self._make_request("GET", "equity/portfolio")
    
    def get_position(self, ticker: str) -> Dict:
        """
        Get a specific position by ticker.
        
        Args:
            ticker: Instrument ticker
            
        Returns:
            Position details
        """
        return self._make_request("GET", f"equity/portfolio/{ticker}")
    
    def get_historical_orders(self, ticker: Optional[str] = None, limit: int = 50) -> Dict:
        """
        Get historical order data.
        
        Args:
            ticker: Filter by instrument ticker (optional)
            limit: Maximum number of items to return (max 50)
            
        Returns:
            Dictionary containing order history and pagination info
        """
        params = {}
        if ticker:
            params["ticker"] = ticker
        if limit:
            params["limit"] = min(limit, 50)  # API limit is 50
            
        return self._make_request("GET", "equity/history/orders", params=params)
    
    def get_dividends(self) -> Dict:
        """
        Get paid out dividends.
        
        Returns:
            Dictionary containing dividend data
        """
        return self._make_request("GET", "history/dividends")
    
    def get_transactions(self) -> Dict:
        """
        Get transaction list.
        
        Returns:
            Dictionary containing transaction data
        """
        return self._make_request("GET", "history/transactions")
    
    def export_data(self, export_type: str) -> str:
        """
        Request a data export.
        
        Args:
            export_type: Type of export (e.g., "orders", "transactions")
            
        Returns:
            Export ID
        """
        data = {"type": export_type}
        return self._make_request("POST", "history/exports", data=data)
    
    def get_exports_list(self) -> List[Dict]:
        """
        Get list of available exports.
        
        Returns:
            List of exports
        """
        return self._make_request("GET", "history/exports")
    
    def download_export(self, export_id: str) -> str:
        """
        Download a specific export.
        
        Args:
            export_id: Export ID
            
        Returns:
            CSV content as string
        """
        url = f"{self.BASE_URL}/equity/history/exports/{export_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.text
