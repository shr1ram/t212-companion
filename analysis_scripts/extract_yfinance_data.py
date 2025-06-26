#!/usr/bin/env python3
"""Extract historical price data from Yahoo Finance and save to JSON files."""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yfinance_companion.api import YFinanceAPI
from t212_companion.utils import load_from_json, save_to_json


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract historical price data from Yahoo Finance')
    parser.add_argument('--positions-file', type=str, default=None,
                        help='Path to positions data file (CSV or JSON)')
    parser.add_argument('--data-dir', type=str, default='t212_data',
                        help='Directory containing T212 data files')
    parser.add_argument('--output-dir', type=str, default='yfinance_data',
                        help='Directory to save output files')
    parser.add_argument('--period', type=str, default='max',
                        help='Time period for historical data (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (e.g., 1d, 1wk, 1mo)')
    return parser.parse_args()


def load_positions(args):
    """Load positions data from file."""
    if args.positions_file:
        positions_file = args.positions_file
    else:
        # Find positions file in the data directory
        positions_file_name = 'positions.json'
        positions_files = [f for f in os.listdir(args.data_dir) if f == positions_file_name]
        if not positions_files:
            raise FileNotFoundError(f"No positions file found in {args.data_dir} directory")
        positions_file = os.path.join(args.data_dir, sorted(positions_files)[-1])
    
    print(f"Loading positions from {positions_file}")
    
    # Load data based on file extension
    if positions_file.endswith('.json'):
        positions = load_from_json(positions_file)
    else:
        positions = pd.read_csv(positions_file).to_dict('records')
    
    print(f"Loaded {len(positions)} positions")
    return positions


def main():
    """Main function."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        # Load positions data
        positions = load_positions(args)
        
        # Initialize YFinance API
        yf_api = YFinanceAPI()
        
        # Extract tickers from positions
        tickers = [position['ticker'] for position in positions]
        
        # Get timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Load historical data for each ticker
        print(f"Loading historical data for period: {args.period}, interval: {args.interval}")
        historical_data = {}
        
        for ticker in tickers:
            print(f"Getting data for {ticker}...")
            yahoo_ticker = yf_api.map_ticker(ticker)
            print(f"  - Mapped to Yahoo Finance ticker: {yahoo_ticker}")
            
            # Get historical data
            data = yf_api.get_historical_data(ticker, period=args.period, interval=args.interval)
            
            if data is not None and not data.empty:
                # Convert DataFrame to dictionary for JSON serialization
                # First reset index to make date a column
                data_dict = data.reset_index().to_dict(orient='records')
                historical_data[ticker] = data_dict
                print(f"  - Got {len(data_dict)} data points")
            else:
                print(f"  - No data found for {ticker}")
        
        # Save each instrument to a separate JSON file
        for ticker, data in historical_data.items():
            # Create a safe filename from ticker
            safe_ticker = ticker.replace('.', '_').replace('/', '_')
            output_file = os.path.join(args.output_dir, f"{safe_ticker}.json")
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4, default=str)
            print(f"Historical data for {ticker} saved to {output_file}")
        
        # Save last update timestamp to a text file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(args.output_dir, "last_update.txt"), 'w') as f:
            f.write(f"Last updated: {timestamp}")
        
        print(f"All historical data saved to {args.output_dir} directory")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
