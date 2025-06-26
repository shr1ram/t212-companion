#!/usr/bin/env python3
"""
Analyze backtest data and generate performance reports.
This script loads historical position data and price data to perform
statistical analysis and generate performance charts.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_analysis import PortfolioAnalyzer
from t212_companion.utils import load_from_json, load_from_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    parser.add_argument('--ignore-warnings', action='store_true',
                        help='Ignore date mismatch warnings and proceed with analysis')
    return parser.parse_args()


def check_last_update_dates(args):
    """
    Check the last update dates for different data sources and warn if they're out of sync.
    
    Returns:
        dict: A dictionary with data source names as keys and datetime objects as values
    """
    update_dates = {}
    
    # Check each data directory for last_update.txt
    for dir_name, dir_path in [
        ('backtest_data', args.backtest_data_dir),
        ('yfinance_data', args.yfinance_data_dir),
        ('reports', args.output_dir)
    ]:
        last_update_file = os.path.join(dir_path, "last_update.txt")
        if os.path.exists(last_update_file):
            with open(last_update_file, 'r') as f:
                content = f.read().strip()
                # Extract datetime from "Last updated: YYYY-MM-DD HH:MM:SS" format
                if "Last updated:" in content:
                    date_str = content.replace("Last updated:", "").strip()
                    try:
                        update_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                        update_dates[dir_name] = update_date
                    except ValueError:
                        logger.warning(f"Could not parse date from {last_update_file}: {content}")
    
    # Check for mismatches
    if len(update_dates) > 1:
        dates_list = list(update_dates.items())
        has_warning = False
        
        for i in range(len(dates_list)):
            for j in range(i+1, len(dates_list)):
                source1, date1 = dates_list[i]
                source2, date2 = dates_list[j]
                
                # Calculate difference in days
                diff = abs((date1 - date2).total_seconds() / 86400)
                
                if diff > 1:  # More than 1 day difference
                    logger.warning(f"DATA MISMATCH WARNING: {source1} was last updated on {date1.strftime('%Y-%m-%d')} "
                                  f"but {source2} was last updated on {date2.strftime('%Y-%m-%d')} "
                                  f"({diff:.1f} days difference)")
                    has_warning = True
        
        if has_warning and not args.ignore_warnings:
            logger.warning("Data sources are out of sync. This may lead to incomplete or misleading analysis.")
            logger.warning("Consider updating all data sources to the same date.")
            logger.warning("Use --ignore-warnings to proceed anyway.")
            
    return update_dates


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


def combine_position_and_price_data(backtest_data, historical_prices, args):
    """
    Combine position data with price data to create a complete dataset
    for analysis with position sizes at each date.
    """
    print("Combining position and price data")
    
    combined_data = {}
    
    # Track the overall date range of data
    overall_min_date = None
    overall_max_date = None
    date_mismatches = []
    
    for ticker in backtest_data.keys():
        if ticker not in historical_prices:
            print(f"  - Warning: No price data found for {ticker}")
            continue
        
        position_df = backtest_data[ticker]
        price_df = historical_prices[ticker]
        
        # Determine the date ranges
        if not position_df.empty and not price_df.empty:
            position_min = position_df.index.min()
            position_max = position_df.index.max()
            price_min = price_df.index.min()
            price_max = price_df.index.max()
            
            # Calculate the common date range
            start_date = max(position_min, price_min)
            end_date = min(position_max, price_max)
            
            # Track overall date range
            if overall_min_date is None or start_date < overall_min_date:
                overall_min_date = start_date
            if overall_max_date is None or end_date > overall_max_date:
                overall_max_date = end_date
            
            # Check for any date mismatches (even 1 day)
            position_price_diff = abs((position_max - price_max).days)
            if position_price_diff > 0:
                date_mismatches.append({
                    'ticker': ticker,
                    'position_range': f"{position_min.date()} to {position_max.date()}",
                    'price_range': f"{price_min.date()} to {price_max.date()}",
                    'difference_days': position_price_diff
                })
                logger.warning(f"DATE MISMATCH: {ticker} position data ends on {position_max.date()}, "
                              f"but price data ends on {price_max.date()} "
                              f"({position_price_diff} days difference)")
        
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
    
    # Check for end date mismatches between instruments
    instrument_end_dates = {}
    for ticker, df in combined_data.items():
        if not df.empty:
            instrument_end_dates[ticker] = df.index.max()
    
    # Find the latest end date across all instruments
    if instrument_end_dates:
        latest_date = max(instrument_end_dates.values())
        instruments_not_up_to_date = []
        
        # Check which instruments don't have data up to the latest date
        for ticker, end_date in instrument_end_dates.items():
            if end_date != latest_date:
                diff_days = (latest_date - end_date).days
                instruments_not_up_to_date.append({
                    'ticker': ticker,
                    'end_date': end_date.date(),
                    'latest_date': latest_date.date(),
                    'days_behind': diff_days
                })
                logger.warning(f"INSTRUMENT DATE MISMATCH: {ticker} data ends on {end_date.date()}, "
                              f"which is {diff_days} days behind the latest data ({latest_date.date()})")
        
        # Add to date mismatches list
        if instruments_not_up_to_date and not date_mismatches:
            date_mismatches = []
        date_mismatches.extend(instruments_not_up_to_date)
    
    # Print summary of date range used for analysis
    if overall_min_date and overall_max_date:
        print(f"\nAnalysis will use data from {overall_min_date.date()} to {overall_max_date.date()}")
        
    # Print summary of date mismatches if any
    if date_mismatches:
        print("\nWARNING: Date mismatches detected:")
        for mismatch in date_mismatches:
            if 'position_range' in mismatch:
                print(f"  - {mismatch['ticker']}: Position data: {mismatch['position_range']}, "
                      f"Price data: {mismatch['price_range']} "
                      f"({mismatch['difference_days']} days difference)")
            elif 'days_behind' in mismatch:
                print(f"  - {mismatch['ticker']}: Data ends on {mismatch['end_date']}, "
                      f"which is {mismatch['days_behind']} days behind the latest data ({mismatch['latest_date']})")
        print("\nThis may lead to incomplete or misleading analysis results.")
        print("Use --ignore-warnings to proceed with analysis despite these warnings.")
        
        # If warnings are not ignored, exit
        if not args.ignore_warnings:
            logger.error("Analysis aborted due to date mismatches. Use --ignore-warnings to proceed anyway.")
            sys.exit(1)
    
    return combined_data


def main():
    """Main function."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        # Check last update dates for data sources
        update_dates = check_last_update_dates(args)
        if update_dates:
            print("\nData source last update dates:")
            for source, date in update_dates.items():
                print(f"  - {source}: {date.strftime('%Y-%m-%d %H:%M:%S')}")
            print()  # Add a blank line
        
        # Load data
        backtest_data = load_backtest_data(args)
        historical_prices = load_historical_prices(args)
        
        # Combine position and price data
        combined_data = combine_position_and_price_data(backtest_data, historical_prices, args)
        
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
            "instruments_analyzed": list(combined_data.keys()),
            "data_date_ranges": {
                "analysis_period": {
                    "start": overall_min_date.strftime("%Y-%m-%d") if 'overall_min_date' in locals() else None,
                    "end": overall_max_date.strftime("%Y-%m-%d") if 'overall_max_date' in locals() else None
                },
                "date_mismatches": date_mismatches if 'date_mismatches' in locals() else []
            }
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
