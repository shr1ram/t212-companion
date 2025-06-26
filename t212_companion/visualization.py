"""Data visualization module for portfolio analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
from datetime import datetime

# Set seaborn style
sns.set_style("whitegrid")


class PortfolioVisualizer:
    """Class for visualizing portfolio data."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the PortfolioVisualizer class.
        
        Args:
            output_dir: Directory to save visualizations (optional)
        """
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_portfolio_composition(self, positions: pd.DataFrame, 
                                  title: str = "Portfolio Composition",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a pie chart showing portfolio composition by value.
        
        Args:
            positions: DataFrame containing position data
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Calculate position values
        if 'quantity' in positions.columns and 'currentPrice' in positions.columns:
            positions = positions.copy()
            positions['value'] = positions['quantity'] * positions['currentPrice']
            
            # Get ticker and value for pie chart
            labels = positions['ticker'].tolist()
            values = positions['value'].tolist()
            
            # Create the pie chart
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title(title)
            
            # Save if path is provided
            if save_path or self.output_dir:
                path = save_path or os.path.join(self.output_dir, f"portfolio_composition_{datetime.now().strftime('%Y%m%d')}.png")
                plt.savefig(path, bbox_inches='tight', dpi=300)
            
            return fig
        else:
            raise ValueError("Positions DataFrame must contain 'quantity' and 'currentPrice' columns")
    
    def plot_returns_over_time(self, returns: pd.Series,
                              title: str = "Portfolio Returns Over Time",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cumulative returns over time.
        
        Args:
            returns: Series of daily returns
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod() - 1
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cumulative_returns.index, cumulative_returns.values * 100)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns (%)')
        ax.set_title(title)
        ax.grid(True)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
        
        # Save if path is provided
        if save_path or self.output_dir:
            path = save_path or os.path.join(self.output_dir, f"returns_over_time_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def plot_drawdowns(self, drawdowns: pd.Series,
                      title: str = "Portfolio Drawdowns",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot drawdowns over time.
        
        Args:
            drawdowns: Series of drawdowns
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(drawdowns.index, drawdowns.values * 100, 0, color='red', alpha=0.3)
        ax.plot(drawdowns.index, drawdowns.values * 100, color='red', alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title(title)
        ax.grid(True)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
        
        # Save if path is provided
        if save_path or self.output_dir:
            path = save_path or os.path.join(self.output_dir, f"drawdowns_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def plot_monthly_returns_heatmap(self, returns: pd.Series,
                                    title: str = "Monthly Returns Heatmap",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap of monthly returns.
        
        Args:
            returns: Series of daily returns
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Resample to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create a pivot table with years as rows and months as columns
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first()
        monthly_pivot = monthly_pivot.unstack()
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(monthly_pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn", 
                   linewidths=0.5, ax=ax, cbar_kws={'label': 'Returns (%)'})
        ax.set_title(title)
        ax.set_ylabel('Year')
        ax.set_xlabel('Month')
        
        # Save if path is provided
        if save_path or self.output_dir:
            path = save_path or os.path.join(self.output_dir, f"monthly_returns_heatmap_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def plot_rolling_statistics(self, returns: pd.Series, window: int = 30,
                               title: str = "Rolling Statistics",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot rolling statistics (returns, volatility, Sharpe ratio).
        
        Args:
            returns: Series of daily returns
            window: Rolling window size in days
            title: Plot title
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Calculate rolling statistics
        rolling_return = returns.rolling(window=window).mean() * 252 * 100  # Annualized and in percentage
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized and in percentage
        rolling_sharpe = rolling_return / rolling_vol  # Simplified Sharpe (no risk-free rate)
        
        # Create the plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Plot rolling returns
        axes[0].plot(rolling_return.index, rolling_return.values)
        axes[0].set_ylabel('Ann. Return (%)')
        axes[0].set_title(f'{window}-Day Rolling Annualized Return')
        axes[0].grid(True)
        
        # Plot rolling volatility
        axes[1].plot(rolling_vol.index, rolling_vol.values)
        axes[1].set_ylabel('Ann. Volatility (%)')
        axes[1].set_title(f'{window}-Day Rolling Annualized Volatility')
        axes[1].grid(True)
        
        # Plot rolling Sharpe ratio
        axes[2].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[2].set_ylabel('Sharpe Ratio')
        axes[2].set_title(f'{window}-Day Rolling Sharpe Ratio')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path or self.output_dir:
            path = save_path or os.path.join(self.output_dir, f"rolling_stats_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def create_performance_dashboard(self, stats_report: Dict, returns: pd.Series, drawdowns: pd.Series,
                                    positions: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive performance dashboard.
        
        Args:
            stats_report: Dictionary containing portfolio statistics
            returns: Series of daily returns
            drawdowns: Series of drawdowns
            positions: DataFrame containing position data
            save_path: Path to save the figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Define grid layout
        gs = fig.add_gridspec(3, 3)
        
        # Plot cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        cumulative_returns = (1 + returns).cumprod() - 1
        ax1.plot(cumulative_returns.index, cumulative_returns.values * 100)
        ax1.set_title('Cumulative Returns (%)')
        ax1.grid(True)
        
        # Plot drawdowns
        ax2 = fig.add_subplot(gs[1, :])
        ax2.fill_between(drawdowns.index, drawdowns.values * 100, 0, color='red', alpha=0.3)
        ax2.plot(drawdowns.index, drawdowns.values * 100, color='red', alpha=0.5)
        ax2.set_title('Drawdowns (%)')
        ax2.grid(True)
        
        # Plot portfolio composition
        ax3 = fig.add_subplot(gs[2, 0])
        if 'quantity' in positions.columns and 'currentPrice' in positions.columns:
            positions = positions.copy()
            positions['value'] = positions['quantity'] * positions['currentPrice']
            labels = positions['ticker'].tolist()
            values = positions['value'].tolist()
            ax3.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax3.axis('equal')
            ax3.set_title('Portfolio Composition')
        
        # Plot key statistics
        ax4 = fig.add_subplot(gs[2, 1:])
        stats = [
            f"Portfolio Value: ${stats_report['portfolio_value']:,.2f}",
            f"Sharpe Ratio: {stats_report['risk_metrics']['sharpe_ratio']:.2f}",
            f"Sortino Ratio: {stats_report['risk_metrics']['sortino_ratio']:.2f}",
            f"Volatility: {stats_report['risk_metrics']['volatility']*100:.2f}%",
            f"Max Drawdown: {stats_report['risk_metrics']['max_drawdown']*100:.2f}%",
            f"Daily Return: {stats_report['returns']['daily']*100:.2f}%",
            f"Monthly Return: {stats_report['returns']['monthly']*100:.2f}%",
            f"Yearly Return: {stats_report['returns']['yearly']*100:.2f}%"
        ]
        ax4.axis('off')
        y_pos = 0.9
        for stat in stats:
            ax4.text(0.1, y_pos, stat, fontsize=12)
            y_pos -= 0.1
        ax4.set_title('Key Statistics')
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path or self.output_dir:
            path = save_path or os.path.join(self.output_dir, f"performance_dashboard_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(path, bbox_inches='tight', dpi=300)
        
        return fig
