#!/usr/bin/env python3
"""
Analyze backtest data and generate performance reports.
This script loads historical position data and price data to perform
statistical analysis and generate performance charts.
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yfinance_companion.api import PortfolioAnalyzer
from t212_companion.utils import load_from_json, load_from_csv


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze backtest data and generate performance reports')
    parser.add_argument('--backtest-data-dir', type=str, default='backtest_data',
                        help='Directory containing backtest data files')
    parser.add_argument('--yfinance-data-dir', type=str, default='yfinance_data',
                        help='Directory containing Yahoo Finance data files')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Directory to save output files')
    parser.add_argument('--risk-free-rate', type=float, default=0,
                        help='Annual risk-free rate for calculations')
    parser.add_argument('--ticker', type=str, default=None,
                        help='Analyze a specific ticker only')
    return parser.parse_args()


def load_backtest_data(args):
    """Load historical position data from backtest data directory."""
    print(f"Loading backtest data from {args.backtest_data_dir} directory")
    
    # Check if the directory exists
    if not os.path.exists(args.backtest_data_dir):
        raise FileNotFoundError(f"Directory not found: {args.backtest_data_dir}")
    
    # Get all JSON files (except last_update.txt)
    if args.ticker:
        # If a specific ticker is provided, only load that file
        safe_ticker = args.ticker.replace('.', '_').replace('/', '_')
        instrument_files = [f"{safe_ticker}.json"]
    else:
        # Otherwise load all JSON files
        instrument_files = [f for f in os.listdir(args.backtest_data_dir) 
                           if f.endswith('.json') and f != 'last_update.txt']
    
    if not instrument_files:
        raise FileNotFoundError(f"No backtest data found in {args.backtest_data_dir} directory")
    
    # Load each instrument file
    backtest_data = {}
    for file in instrument_files:
        ticker = file.replace('.json', '')
        file_path = os.path.join(args.backtest_data_dir, file)
        
        # Skip if file doesn't exist (might happen if a specific ticker was requested)
        if not os.path.exists(file_path):
            print(f"  - File not found: {file_path}")
            continue
            
        print(f"  - Loading position data for {ticker}")
        
        # Load position data
        position_data = load_from_json(file_path)
        
        # Convert to DataFrame
        df = pd.DataFrame(position_data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        backtest_data[ticker] = df
    
    print(f"Loaded backtest data for {len(backtest_data)} instruments")
    return backtest_data


def load_historical_prices(args):
    """Load historical price data from individual instrument files."""
    print(f"Loading historical price data from {args.yfinance_data_dir} directory")
    
    # Check if the directory exists
    if not os.path.exists(args.yfinance_data_dir):
        raise FileNotFoundError(f"Directory not found: {args.yfinance_data_dir}")
    
    # Get all JSON files (except last_update.txt)
    if args.ticker:
        # If a specific ticker is provided, only load that file
        safe_ticker = args.ticker.replace('.', '_').replace('/', '_')
        instrument_files = [f"{safe_ticker}.json"]
    else:
        # Otherwise load all JSON files
        instrument_files = [f for f in os.listdir(args.yfinance_data_dir) 
                           if f.endswith('.json') and f != 'last_update.txt']
    
    if not instrument_files:
        raise FileNotFoundError(f"No historical price data found in {args.yfinance_data_dir} directory")
    
    # Load each instrument file
    historical_prices = {}
    for file in instrument_files:
        ticker = file.replace('.json', '')
        file_path = os.path.join(args.yfinance_data_dir, file)
        
        # Skip if file doesn't exist (might happen if a specific ticker was requested)
        if not os.path.exists(file_path):
            print(f"  - File not found: {file_path}")
            continue
            
        print(f"  - Loading price data for {ticker}")
        
        data = load_from_json(file_path)
        
        # Convert to DataFrame and set Date as index
        df = pd.DataFrame(data)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        historical_prices[ticker] = df
    
    print(f"Loaded historical price data for {len(historical_prices)} instruments")
    return historical_prices


def combine_position_and_price_data(backtest_data, historical_prices):
    """
    Combine position data with price data to create a complete dataset
    for analysis with position sizes at each date.
    """
    print("Combining position and price data")
    
    combined_data = {}
    
    for ticker in backtest_data.keys():
        if ticker not in historical_prices:
            print(f"  - Warning: No price data found for {ticker}")
            continue
        
        position_df = backtest_data[ticker]
        price_df = historical_prices[ticker]
        
        # Merge position and price data
        combined_df = pd.DataFrame(index=price_df.index)
        
        # Add price data - IMPORTANT: Use the column names that PortfolioAnalyzer expects
        price_col = 'Adj Close' if 'Adj Close' in price_df.columns else 'Close'
        # Copy the price data to both the original column name AND to our working 'price' column
        combined_df[price_col] = price_df[price_col]  # This is what PortfolioAnalyzer will look for
        combined_df['price'] = price_df[price_col]    # This is what we'll use for our calculations
        
        # Add position data (quantity)
        # Reindex position data to match price data dates
        if not position_df.empty:
            # Forward fill position quantities to have a quantity for each price date
            position_series = position_df['quantity'].reindex(
                combined_df.index, method='ffill').fillna(0)
            combined_df['quantity'] = position_series
        else:
            combined_df['quantity'] = 0
        
        # Calculate position value
        combined_df['value'] = combined_df['price'] * combined_df['quantity']
        
        # Calculate daily returns (for price)
        combined_df['price_return'] = combined_df['price'].pct_change()
        
        # Calculate position returns (accounting for quantity changes)
        combined_df['position_return'] = combined_df['value'].pct_change()
        
        # Store in result dictionary
        combined_data[ticker] = combined_df
    
    return combined_data


def main():
    """Main function."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        # Load data
        backtest_data = load_backtest_data(args)
        historical_prices = load_historical_prices(args)
        
        # Combine position and price data
        combined_data = combine_position_and_price_data(backtest_data, historical_prices)
        
        # Create a portfolio analyzer
        # First, create a positions list for the analyzer in the format it expects
        # PortfolioAnalyzer expects a list of dictionaries with 'ticker' and 'quantity' keys
        positions_list = []
        for ticker, data in combined_data.items():
            if not data.empty and 'quantity' in data.columns:
                # Use the most recent quantity for each instrument
                latest_quantity = data['quantity'].iloc[-1]
                if latest_quantity > 0:
                    positions_list.append({
                        'ticker': ticker,
                        'quantity': float(latest_quantity)
                    })
        
        print(f"Created positions list with {len(positions_list)} instruments")
        
        # Create the analyzer with positions
        analyzer = PortfolioAnalyzer(positions=positions_list, risk_free_rate=args.risk_free_rate)
        
        # Set historical data for the analyzer
        analyzer.historical_data = {ticker: data for ticker, data in combined_data.items()}
        
        # Debug: Print what data we actually have
        print("\nDEBUG: Combined Data Summary:")
        for ticker, data in combined_data.items():
            print(f"Ticker: {ticker}")
            if data.empty:
                print(f"  - No data available")
            else:
                print(f"  - Data shape: {data.shape}")
                print(f"  - Date range: {data.index.min()} to {data.index.max()}")
                print(f"  - Columns: {list(data.columns)}")
                
                # Show first few rows of position value
                if 'value' in data.columns:
                    print(f"  - First few rows of position value:")
                    print(data['value'].head())
                else:
                    print("  - No position value column found")
            print()
        
        # Calculate portfolio returns
        portfolio_returns = analyzer.calculate_portfolio_returns()
        
        # Calculate Sharpe ratios
        sharpe_ratios = analyzer.calculate_all_sharpe_ratios()
        
        # Print Sharpe ratios
        print("\nSharpe Ratios:")
        for ticker, sharpe in sharpe_ratios.items():
            print(f"{ticker}: {sharpe:.4f}")
        
        # Plot prices
        print("\nGenerating price chart...")
        price_fig = analyzer.plot_prices()
        if price_fig:
            price_fig.tight_layout()
            price_fig_path = os.path.join(args.output_dir, "price_performance.png")
            price_fig.savefig(price_fig_path)
            print(f"Price chart saved to {price_fig_path}")
        
        # Plot portfolio performance
        print("\nGenerating portfolio performance chart...")
        portfolio_fig = analyzer.plot_portfolio_performance()
        if portfolio_fig:
            portfolio_fig.tight_layout()
            portfolio_fig_path = os.path.join(args.output_dir, "portfolio_performance.png")
            portfolio_fig.savefig(portfolio_fig_path)
            print(f"Portfolio performance chart saved to {portfolio_fig_path}")
        
        # Plot Sharpe ratios
        print("\nGenerating Sharpe ratio chart...")
        sharpe_fig = analyzer.plot_sharpe_ratios()
        if sharpe_fig:
            sharpe_fig.tight_layout()
            sharpe_fig_path = os.path.join(args.output_dir, "sharpe_ratios.png")
            sharpe_fig.savefig(sharpe_fig_path)
            print(f"Sharpe ratio chart saved to {sharpe_fig_path}")
        
        # Get portfolio Sharpe ratio (if available in the results)
        portfolio_sharpe = None
        if 'Portfolio' in sharpe_ratios:
            portfolio_sharpe = sharpe_ratios['Portfolio']
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save results to JSON
        results = {
            "timestamp": timestamp,
            "risk_free_rate": args.risk_free_rate,
            "sharpe_ratios": {
                "portfolio": portfolio_sharpe,
                "instruments": {ticker: sharpe for ticker, sharpe in sharpe_ratios.items() if ticker != 'Portfolio'}
            },
            "instruments_analyzed": list(combined_data.keys())
        }
        results_path = os.path.join(args.output_dir, "backtest_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"\nResults saved to {results_path}")
        
        # Save last update timestamp to a text file
        with open(os.path.join(args.output_dir, "last_update.txt"), 'w') as f:
            f.write(f"Last updated: {timestamp}")
        
        print("\nBacktest analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
