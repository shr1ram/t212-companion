#!/usr/bin/env python3
"""Backtest portfolio performance using saved data from T212 and Yahoo Finance."""

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
    parser = argparse.ArgumentParser(description='Backtest portfolio performance using saved data')
    parser.add_argument('--positions-file', type=str, default=None,
                        help='Path to positions data file (CSV or JSON)')
    parser.add_argument('--historical-file', type=str, default=None,
                        help='Path to historical price data file (JSON)')
    parser.add_argument('--t212-data-dir', type=str, default='t212_data',
                        help='Directory containing T212 data files')
    parser.add_argument('--yfinance-data-dir', type=str, default='yfinance_data',
                        help='Directory containing Yahoo Finance data files')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Directory to save output files')
    parser.add_argument('--risk-free-rate', type=float, default=0,
                        help='Annual risk-free rate for calculations')
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
    
    return positions


def load_historical_data(args):
    """Load historical price data from individual instrument files."""
    if args.historical_file:
        # If a specific file is provided, load just that file
        historical_file = args.historical_file
        print(f"Loading historical price data from {historical_file}")
        data = load_from_json(historical_file)
        ticker = os.path.basename(historical_file).replace('.json', '')
        
        # Process the data
        df = pd.DataFrame(data)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        processed_data = {ticker: df}
    else:
        # Load all instrument files from the YFinance data directory
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
        processed_data = {}
        for file in instrument_files:
            ticker = file.replace('.json', '')
            file_path = os.path.join(args.yfinance_data_dir, file)
            print(f"  - Loading data for {ticker}")
            data = load_from_json(file_path)
            
            # Process the data
            if data:  # Skip empty data
                # Convert list of dictionaries to DataFrame
                df = pd.DataFrame(data)
                
                # Convert Date column to datetime and set as index
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                
                processed_data[ticker] = df
    
    return processed_data


def main():
    """Main function."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        # Load positions data
        positions = load_positions(args)
        print(f"Loaded {len(positions)} positions")
        
        # Load historical data
        historical_data = load_historical_data(args)
        print(f"Loaded historical data for {len(historical_data)} instruments")
        
        # Initialize portfolio analyzer with loaded data
        analyzer = PortfolioAnalyzer(positions, risk_free_rate=args.risk_free_rate)
        analyzer.historical_data = historical_data  # Set historical data directly
        
        # Debug: Print what data we actually received
        print("\nDEBUG: Historical Data Summary:")
        for ticker, data in analyzer.historical_data.items():
            print(f"Ticker: {ticker}")
            if data.empty:
                print(f"  - No data available")
            else:
                print(f"  - Data shape: {data.shape}")
                print(f"  - Date range: {data.index.min()} to {data.index.max()}")
                print(f"  - Columns: {list(data.columns)}")
                
                # Show first few rows of price data
                price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                if price_col in data.columns:
                    print(f"  - First few rows of {price_col} data:")
                    print(data[price_col].head())
                else:
                    print("  - No price column found")
            print()
        
        # Calculate portfolio returns
        portfolio_returns = analyzer.calculate_portfolio_returns()
        
        # Calculate Sharpe ratios
        sharpe_ratios = analyzer.calculate_all_sharpe_ratios()
        
        # Print Sharpe ratios
        print("\nSharpe Ratios:")
        for ticker, sharpe in sharpe_ratios.items():
            print(f"{ticker}: {sharpe:.4f}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
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
            "positions_count": len(positions),
            "instruments": list(analyzer.historical_data.keys())
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
