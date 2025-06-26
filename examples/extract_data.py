#!/usr/bin/env python3
"""
Example script to extract portfolio data from Trading 212 API.
"""

import os
import sys
import json
from datetime import datetime
import argparse

# Add parent directory to path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t212_companion.api import T212API
from t212_companion.utils import save_to_json, save_to_csv


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract portfolio data from Trading 212 API')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Directory to save output files')
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='json',
                        help='Output file format')
    parser.add_argument('--ticker', type=str, help='Filter by specific ticker')
    return parser.parse_args()


def main():
    """Main function to extract data from Trading 212 API."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        # Initialize API client
        api = T212API()
        
        # Extract account data
        print("Extracting account data...")
        account_data = api.get_account_data()
        
        # Extract portfolio positions
        print("Extracting portfolio positions...")
        positions = api.get_positions()
        
        # Extract historical orders
        print("Extracting historical orders...")
        historical_orders = api.get_historical_orders(ticker=args.ticker)
        
        # Extract dividends
        print("Extracting dividends...")
        dividends = api.get_dividends()
        
        # Extract transactions
        print("Extracting transactions...")
        transactions = api.get_transactions()
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.format == 'json':
            # Save to JSON files
            save_to_json(account_data, f"account_data", args.output_dir)
            save_to_json(positions, f"positions", args.output_dir)
            save_to_json(historical_orders, f"historical_orders", args.output_dir)
            save_to_json(dividends, f"dividends", args.output_dir)
            save_to_json(transactions, f"transactions", args.output_dir)
        else:
            # Save to CSV files
            save_to_csv(account_data, f"account_data", args.output_dir)
            save_to_csv(positions, f"positions", args.output_dir)
            save_to_csv(historical_orders['items'] if 'items' in historical_orders else historical_orders, 
                       f"historical_orders", args.output_dir)
            save_to_csv(dividends['items'] if 'items' in dividends else dividends, 
                       f"dividends", args.output_dir)
            save_to_csv(transactions['items'] if 'items' in transactions else transactions, 
                       f"transactions", args.output_dir)
        
        print(f"Data extraction complete. Files saved to {args.output_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
