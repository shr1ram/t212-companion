#!/usr/bin/env python3
"""
Run the complete data extraction and backtest pipeline.

This script runs the following scripts in order:
1. extract_t212_data.py - Extract Trading 212 data
2. extract_yfinance_data.py - Extract Yahoo Finance data
3. backtest_data.py - Generate backtest data

Optionally, it can also run analyse_backtest.py to analyze the results.
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_script(script_path, args=None):
    """
    Run a Python script with the given arguments.
    
    Args:
        script_path: Path to the script to run
        args: List of arguments to pass to the script
        
    Returns:
        True if the script ran successfully, False otherwise
    """
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
        
    script_name = os.path.basename(script_path)
    logger.info(f"Running {script_name}...")
    
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"{script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{script_name} failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the complete data extraction and backtest pipeline")
    parser.add_argument("--analyze", action="store_true", help="Also run analyse_backtest.py after generating data")
    parser.add_argument("--ignore-warnings", action="store_true", help="Pass --ignore-warnings to analyse_backtest.py")
    parser.add_argument("--clean", action="store_true", help="Clean data directories before running the pipeline")
    args = parser.parse_args()
    
    # Record start time
    start_time = datetime.now()
    logger.info(f"Pipeline started at {start_time}")
    
    # Clean data directories if requested
    if args.clean:
        logger.info("Cleaning data directories...")
        clean_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_data.py")
        if not run_script(clean_script, ["--confirm"]):
            logger.error("Failed to clean data directories. Aborting pipeline.")
            return
    
    # Define scripts to run in order
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_scripts")
    scripts = [
        os.path.join(scripts_dir, "extract_t212_data.py"),
        os.path.join(scripts_dir, "extract_yfinance_data.py"),
        os.path.join(scripts_dir, "backtest_data.py")
    ]
    
    # Run each script in order
    for script in scripts:
        if not os.path.exists(script):
            logger.error(f"Script not found: {script}")
            return
            
        if not run_script(script):
            logger.error(f"Pipeline failed at {os.path.basename(script)}")
            return
    
    # Run analysis script if requested
    if args.analyze:
        analyze_script = os.path.join(scripts_dir, "analyse_backtest.py")
        analyze_args = ["--ignore-warnings"] if args.ignore_warnings else []
        
        if not run_script(analyze_script, analyze_args):
            logger.error("Analysis failed")
            return
    
    # Record end time and calculate duration
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Pipeline completed at {end_time}")
    logger.info(f"Total duration: {duration}")
    logger.info("All scripts completed successfully")

if __name__ == "__main__":
    main()
