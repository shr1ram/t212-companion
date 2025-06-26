#!/usr/bin/env python3
"""
Clean data directories by removing all files within them.
This script empties the following directories:
- backtest_data
- reports
- t212_data
- yfinance_data
"""

import os
import shutil
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def clean_directory(directory_path):
    """
    Remove all files in a directory without deleting the directory itself.
    
    Args:
        directory_path: Path to the directory to clean
    """
    if not os.path.exists(directory_path):
        logger.info(f"Creating directory: {directory_path}")
        os.makedirs(directory_path)
        return
        
    logger.info(f"Cleaning directory: {directory_path}")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                logger.debug(f"Removed file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                logger.debug(f"Removed directory: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete {file_path}. Reason: {e}")

def main():
    """Main function to clean data directories."""
    parser = argparse.ArgumentParser(description="Clean data directories")
    parser.add_argument("--confirm", action="store_true", help="Confirm deletion without prompting")
    args = parser.parse_args()
    
    # Directories to clean
    directories = [
        "backtest_data",
        "reports",
        "t212_data",
        "yfinance_data"
    ]
    
    # Get confirmation unless --confirm flag is used
    if not args.confirm:
        print("This will delete all files in the following directories:")
        for directory in directories:
            print(f"- {directory}")
        confirmation = input("Are you sure you want to proceed? (y/n): ")
        if confirmation.lower() != 'y':
            logger.info("Operation cancelled by user")
            return
    
    # Clean each directory
    for directory in directories:
        clean_directory(directory)
    
    logger.info("All directories cleaned successfully")

if __name__ == "__main__":
    main()
