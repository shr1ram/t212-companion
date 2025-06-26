# Trading 212 Companion

A set of Python scripts to extract portfolio data from Trading 212 via its API and calculate investment statistics.

## Features

- Extract portfolio data from Trading 212 API
- Calculate investment statistics:
  - Sharpe ratio (backtested and achieved)
  - Drawdowns
  - Volatility
  - Returns (daily, monthly, annual)
  - Maximum drawdown
  - Sortino ratio
  - Beta and Alpha
- Visualize portfolio performance

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your Trading 212 API key:
   ```
   T212_API_KEY=your_api_key_here
   ```

## Usage

### Extract Portfolio Data

```python
from t212_companion.api import T212API

# Initialize the API client
api = T212API()

# Get portfolio positions
positions = api.get_positions()

# Get account data
account = api.get_account_data()

# Get historical orders
orders = api.get_historical_orders()
```

### Calculate Statistics

```python
from t212_companion.stats import PortfolioStats

# Initialize with portfolio data
stats = PortfolioStats(positions, orders)

# Calculate Sharpe ratio
sharpe = stats.calculate_sharpe_ratio()

# Calculate drawdowns
drawdowns = stats.calculate_drawdowns()

# Generate performance report
report = stats.generate_report()
```

## Project Structure

```
t212-companion/
├── t212_companion/
│   ├── __init__.py
│   ├── api.py           # Trading 212 API client
│   ├── stats.py         # Statistical calculations
│   ├── visualization.py # Data visualization
│   └── utils.py         # Utility functions
├── examples/
│   ├── extract_data.py
│   └── calculate_stats.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_stats.py
├── .env                 # API key (not tracked by git)
├── requirements.txt
└── README.md
```

## License

MIT
