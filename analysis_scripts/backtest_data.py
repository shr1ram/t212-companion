#!/usr/bin/env python3
"""
Create historical position data files for backtesting.
This script creates JSON files for each instrument with historical position sizes.
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t212_companion.utils import load_from_json, load_from_csv, save_to_json


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create historical position data files for backtesting')
    parser.add_argument('--positions-file', type=str, default=None,
                        help='Path to positions data file (CSV or JSON)')
    parser.add_argument('--historical-orders-file', type=str, default=None,
                        help='Path to historical orders data file (JSON)')
    parser.add_argument('--t212-data-dir', type=str, default='t212_data',
                        help='Directory containing T212 data files')
    parser.add_argument('--yfinance-data-dir', type=str, default='yfinance_data',
                        help='Directory containing Yahoo Finance data files')
    parser.add_argument('--output-dir', type=str, default='backtest_data',
                        help='Directory to save output files')
    return parser.parse_args()


def load_positions(args):
    """Load positions data from file."""
    if args.positions_file:
        positions_file = args.positions_file
    else:
        # Use the standard positions file name without timestamp
        positions_file = os.path.join(args.t212_data_dir, "positions.json")
        
        # If that doesn't exist, look for CSV version
        if not os.path.exists(positions_file):
            positions_file = os.path.join(args.t212_data_dir, "positions.csv")
            
        # If neither exists, raise an error
        if not os.path.exists(positions_file):
            raise FileNotFoundError(f"No positions file found in {args.t212_data_dir} directory")
    
    print(f"Loading positions from {positions_file}")
    
    # Load data based on file extension
    if positions_file.endswith('.json'):
        positions = load_from_json(positions_file)
    else:
        positions = pd.read_csv(positions_file).to_dict('records')
    
    print(f"Loaded {len(positions)} positions")
    return positions


def load_historical_orders(args):
    """Load historical orders data from file."""
    if args.historical_orders_file:
        orders_file = args.historical_orders_file
    else:
        # Use the standard historical orders file name without timestamp
        orders_file = os.path.join(args.t212_data_dir, "historical_orders.json")
            
        # If file doesn't exist, raise an error
        if not os.path.exists(orders_file):
            raise FileNotFoundError(f"No historical orders file found in {args.t212_data_dir} directory")
    
    print(f"Loading historical orders from {orders_file}")
    
    # Load data
    orders = load_from_json(orders_file)
    
    print(f"Loaded {len(orders)} historical orders")
    return orders


def load_historical_prices(args):
    """Load historical price data from individual instrument files."""
    print(f"Loading historical price data from {args.yfinance_data_dir} directory")
    
    # Check if the directory exists
    if not os.path.exists(args.yfinance_data_dir):
        raise FileNotFoundError(f"Directory not found: {args.yfinance_data_dir}")
    
    # Get all JSON files (except last_update.txt)
    instrument_files = [f for f in os.listdir(args.yfinance_data_dir) 
                       if f.endswith('.json') and f != 'last_update.txt']
    
    if not instrument_files:
        raise FileNotFoundError(f"No historical price data found in {args.yfinance_data_dir} directory")
    
    # Load each instrument file
    historical_prices = {}
    for file in instrument_files:
        ticker = file.replace('.json', '')
        print(f"  - Loading data for {ticker}")
        
        file_path = os.path.join(args.yfinance_data_dir, file)
        data = load_from_json(file_path)
        
        # Convert to DataFrame and set Date as index
        df = pd.DataFrame(data)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        historical_prices[ticker] = df
    
    print(f"Loaded historical price data for {len(historical_prices)} instruments")
    return historical_prices


def create_historical_positions(positions, orders_data, historical_prices):
    """
    Create historical position data for each instrument.
    This combines current positions with historical orders to reconstruct
    position sizes at each date in the historical price data.
    """
    # Extract orders from the orders_data structure
    if isinstance(orders_data, dict) and 'items' in orders_data:
        orders = orders_data['items']
    else:
        orders = orders_data
    
    # Extract unique tickers from positions and orders
    position_tickers = {p['ticker'] for p in positions}
    order_tickers = {o['ticker'] for o in orders if 'ticker' in o}
    all_tickers = position_tickers.union(order_tickers)
    
    print(f"Creating historical position data for {len(all_tickers)} instruments")
    
    # Initialize result dictionary
    historical_positions = {}
    
    for ticker in all_tickers:
        print(f"  - Processing {ticker}")
        
        # Get current position for this ticker
        current_position = next((p for p in positions if p['ticker'] == ticker), None)
        current_quantity = float(current_position['quantity']) if current_position else 0
        
        # Get historical orders for this ticker
        ticker_orders = [o for o in orders if o.get('ticker') == ticker]
        ticker_orders.sort(key=lambda x: x.get('created', ''))  # Sort by creation date
        
        # Get historical price dates for this ticker
        if ticker in historical_prices:
            price_dates = historical_prices[ticker].index
            date_range = pd.date_range(start=price_dates.min(), end=price_dates.max(), freq='D')
        else:
            # If no price data, use order dates
            if ticker_orders:
                order_dates = [pd.to_datetime(o.get('created')) for o in ticker_orders if 'created' in o]
                if order_dates:
                    date_range = pd.date_range(start=min(order_dates), end=max(order_dates), freq='D')
                else:
                    continue  # Skip this ticker if no dates available
            else:
                continue  # Skip this ticker if no orders and no price data
        
        # Create a DataFrame with dates and initialize with current quantity
        position_df = pd.DataFrame(index=date_range)
        position_df['quantity'] = current_quantity
        
        # Work backwards through orders to reconstruct historical positions
        for order in reversed(ticker_orders):
            # Check for date field - could be 'created' or 'dateCreated'
            date_field = next((f for f in ['dateCreated', 'created'] if f in order), None)
            if not date_field:
                continue
                
            # Check for quantity field - could be 'quantity', 'filledQuantity', or 'orderedQuantity'
            quantity = None
            for field in ['filledQuantity', 'orderedQuantity', 'quantity']:
                if field in order and order[field] is not None:
                    quantity = order[field]
                    break
                    
            # If no quantity but we have value and price, calculate quantity
            if quantity is None and 'filledValue' in order and order['filledValue'] is not None:
                # Try to get the price from the order or from historical data
                price = None
                if 'limitPrice' in order and order['limitPrice'] is not None:
                    price = order['limitPrice']
                elif ticker in historical_prices:
                    order_date = pd.to_datetime(order[date_field]).date()
                    price_on_date = historical_prices[ticker].loc[order_date:order_date]
                    if not price_on_date.empty:
                        price_col = 'Adj Close' if 'Adj Close' in price_on_date.columns else 'Close'
                        price = price_on_date[price_col].iloc[0]
                        
                if price and price > 0:
                    quantity = order['filledValue'] / price
            
            # Skip if we couldn't determine quantity
            if quantity is None:
                continue
                
            # Convert quantity to float
            try:
                order_quantity = float(quantity)
            except (ValueError, TypeError):
                continue
                
            order_date = pd.to_datetime(order[date_field]).date()
            
            # Adjust quantity based on order type
            order_type = order.get('type', '').upper()
            if 'BUY' in order_type or order_type == 'MARKET':
                # For buy orders, subtract quantity for dates before the order
                position_df.loc[:order_date, 'quantity'] -= order_quantity
            elif 'SELL' in order_type:
                # For sell orders, add quantity for dates before the order
                position_df.loc[:order_date, 'quantity'] += order_quantity
        
        # Ensure we don't have negative positions (shouldn't happen but just in case)
        position_df['quantity'] = position_df['quantity'].clip(lower=0)
        
        # Convert to list of records with dates
        position_records = []
        for date, row in position_df.iterrows():
            if row['quantity'] > 0:  # Only include dates with non-zero positions
                position_records.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'quantity': row['quantity']
                })
        
        if position_records:
            historical_positions[ticker] = position_records
    
    return historical_positions


def main():
    """Main function."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        # Load data
        positions = load_positions(args)
        orders = load_historical_orders(args)
        historical_prices = load_historical_prices(args)
        
        # Create historical position data
        historical_positions = create_historical_positions(positions, orders, historical_prices)
        
        # Save each instrument's historical position data to a separate file
        print(f"\nSaving historical position data to {args.output_dir} directory")
        for ticker, position_data in historical_positions.items():
            # Create a safe filename from ticker
            safe_ticker = ticker.replace('.', '_').replace('/', '_')
            output_file = os.path.join(args.output_dir, f"{safe_ticker}.json")
            
            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(position_data, f, indent=4, default=str)
            
            print(f"  - Saved {len(position_data)} position records for {ticker}")
        
        # Save last update timestamp to a text file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(args.output_dir, "last_update.txt"), 'w') as f:
            f.write(f"Last updated: {timestamp}")
        
        print(f"\nAll historical position data saved to {args.output_dir} directory")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
