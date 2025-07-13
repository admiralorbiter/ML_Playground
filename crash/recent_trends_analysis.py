"""
Recent Crash Trends Analysis (Last 10 Years)
============================================

This script analyzes crash data from the last 10 years (2015-2025) with detailed
year-by-year trend analysis and predictions for 2025 based on current data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RecentTrendsAnalyzer:
    """
    A focused analyzer for recent crash trends over the last 10 years.
    """
    
    def __init__(self, data_path):
        """Initialize the analyzer with the crash data file path."""
        self.data_path = data_path
        self.df = None
        self.recent_df = None
        self.loaded = False
        
    def load_data(self):
        """Load the crash data and filter for last 10 years."""
        print("Loading crash data...")
        print(f"File path: {self.data_path}")
        
        try:
            # Load with low_memory=False to avoid mixed types warning
            self.df = pd.read_csv(self.data_path, low_memory=False)
            self.loaded = True
            print(f"✓ Data loaded successfully!")
            print(f"Total dataset shape: {self.df.shape}")
            
            # Process date columns
            self._process_date_columns()
            
            # Filter for last 10 years
            self._filter_recent_data()
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
            
        return True
    
    def _process_date_columns(self):
        """Process and convert date columns to proper datetime format."""
        print("\nProcessing date columns...")
        
        # Convert REPORT_DATE to datetime
        if 'REPORT_DATE' in self.df.columns:
            self.df['REPORT_DATE'] = pd.to_datetime(self.df['REPORT_DATE'], format='%Y%m%d', errors='coerce')
            
        # Convert CHANGE_DATE to datetime
        if 'CHANGE_DATE' in self.df.columns:
            self.df['CHANGE_DATE'] = pd.to_datetime(self.df['CHANGE_DATE'], format='%Y%m%d %H%M', errors='coerce')
            
        # Process REPORT_TIME
        if 'REPORT_TIME' in self.df.columns:
            # Convert time to proper format (assuming HHMM format)
            self.df['REPORT_TIME_STR'] = self.df['REPORT_TIME'].astype(str).str.zfill(4)
            self.df['REPORT_HOUR'] = self.df['REPORT_TIME_STR'].str[:2].astype(int, errors='ignore')
            
        print("✓ Date columns processed")
    
    def _filter_recent_data(self):
        """Filter data for the last 10 years (2015-2025)."""
        print("\nFiltering for last 10 years (2015-2025)...")
        
        # Add year column
        self.df['YEAR'] = self.df['REPORT_DATE'].dt.year
        
        # Filter for 2015 onwards
        self.recent_df = self.df[self.df['YEAR'] >= 2015].copy()
        
        print(f"✓ Recent data filtered: {self.recent_df.shape[0]:,} records")
        print(f"Year range: {self.recent_df['YEAR'].min()} to {self.recent_df['YEAR'].max()}")
        
        # Show year distribution
        year_counts = self.recent_df['YEAR'].value_counts().sort_index()
        print("\nRecords by year:")
        for year, count in year_counts.items():
            print(f"  {year}: {count:,} crashes")
    
    def basic_recent_info(self):
        """Display basic information about the recent dataset."""
        if not self.loaded:
            print("Please load data first using load_data()")
            return
            
        print("\n" + "="*60)
        print("RECENT DATA OVERVIEW (2015-2025)")
        print("="*60)
        
        print(f"Recent dataset shape: {self.recent_df.shape}")
        print(f"Memory usage: {self.recent_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Year distribution
        year_dist = self.recent_df['YEAR'].value_counts().sort_index()
        print(f"\nYear distribution:")
        for year, count in year_dist.items():
            print(f"  {year}: {count:,} crashes")
        
        # Basic statistics
        print(f"\nTotal crashes in last 10 years: {len(self.recent_df):,}")
        print(f"Average crashes per year: {len(self.recent_df)/10:,.0f}")
        
    def year_by_year_analysis(self):
        """Detailed year-by-year analysis with trend identification."""
        if not self.loaded:
            print("Please load data first using load_data()")
            return
            
        print("\n" + "="*60)
        print("YEAR-BY-YEAR TREND ANALYSIS")
        print("="*60)
        
        # Calculate year-by-year statistics
        yearly_stats = self.recent_df.groupby('YEAR').agg({
            'FATALITIES': ['sum', 'mean', 'count'],
            'INJURIES': ['sum', 'mean'],
            'VEHICLES_IN_ACCIDENT': ['mean', 'median']
        }).round(2)
        
        # Flatten column names
        yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns]
        yearly_stats = yearly_stats.reset_index()
        
        # Rename columns for clarity
        yearly_stats.columns = [
            'YEAR', 'TOTAL_FATALITIES', 'AVG_FATALITIES_PER_CRASH', 'TOTAL_CRASHES',
            'TOTAL_INJURIES', 'AVG_INJURIES_PER_CRASH', 'AVG_VEHICLES', 'MEDIAN_VEHICLES'
        ]
        
        print("\nYear-by-Year Statistics:")
        print(yearly_stats.to_string(index=False))
        
        # Calculate year-over-year changes
        yearly_stats['CRASH_CHANGE_PCT'] = yearly_stats['TOTAL_CRASHES'].pct_change() * 100
        yearly_stats['FATALITY_CHANGE_PCT'] = yearly_stats['TOTAL_FATALITIES'].pct_change() * 100
        yearly_stats['INJURY_CHANGE_PCT'] = yearly_stats['TOTAL_INJURIES'].pct_change() * 100
        
        print("\nYear-over-Year Percentage Changes:")
        change_cols = ['YEAR', 'CRASH_CHANGE_PCT', 'FATALITY_CHANGE_PCT', 'INJURY_CHANGE_PCT']
        print(yearly_stats[change_cols].round(2).to_string(index=False))
        
        # Identify trends
        self._identify_trends(yearly_stats)
        
        # Visualize trends
        self._plot_yearly_trends(yearly_stats)
        
    def _identify_trends(self, yearly_stats):
        """Identify key trends in the data."""
        print("\n" + "="*60)
        print("TREND IDENTIFICATION")
        print("="*60)
        
        # Overall trends
        first_year = yearly_stats.iloc[0]
        last_complete_year = yearly_stats[yearly_stats['YEAR'] < 2025].iloc[-1]
        
        total_change = ((last_complete_year['TOTAL_CRASHES'] - first_year['TOTAL_CRASHES']) / 
                       first_year['TOTAL_CRASHES']) * 100
        
        fatality_change = ((last_complete_year['TOTAL_FATALITIES'] - first_year['TOTAL_FATALITIES']) / 
                          first_year['TOTAL_FATALITIES']) * 100
        
        injury_change = ((last_complete_year['TOTAL_INJURIES'] - first_year['TOTAL_INJURIES']) / 
                        first_year['TOTAL_INJURIES']) * 100
        
        print(f"\nOverall Changes (2015 to 2024):")
        print(f"  Total crashes: {total_change:+.1f}%")
        print(f"  Total fatalities: {fatality_change:+.1f}%")
        print(f"  Total injuries: {injury_change:+.1f}%")
        
        # Find peak years
        peak_crashes_year = yearly_stats.loc[yearly_stats['TOTAL_CRASHES'].idxmax()]
        peak_fatalities_year = yearly_stats.loc[yearly_stats['TOTAL_FATALITIES'].idxmax()]
        
        print(f"\nPeak Years:")
        print(f"  Most crashes: {peak_crashes_year['YEAR']} ({peak_crashes_year['TOTAL_CRASHES']:,})")
        print(f"  Most fatalities: {peak_fatalities_year['YEAR']} ({peak_fatalities_year['TOTAL_FATALITIES']:,})")
        
        # Recent trends (last 5 years)
        recent_stats = yearly_stats[yearly_stats['YEAR'] >= 2020]
        recent_crash_change = ((recent_stats.iloc[-1]['TOTAL_CRASHES'] - recent_stats.iloc[0]['TOTAL_CRASHES']) / 
                              recent_stats.iloc[0]['TOTAL_CRASHES']) * 100
        
        print(f"\nRecent Trends (2020-2024):")
        print(f"  Crash change: {recent_crash_change:+.1f}%")
        
    def _plot_yearly_trends(self, yearly_stats):
        """Create visualizations for yearly trends."""
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Total crashes over time
        axes[0, 0].plot(yearly_stats['YEAR'], yearly_stats['TOTAL_CRASHES'], 
                        marker='o', linewidth=2, color='blue')
        axes[0, 0].set_title('Total Crashes by Year')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Number of Crashes')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Total fatalities over time
        axes[0, 1].plot(yearly_stats['YEAR'], yearly_stats['TOTAL_FATALITIES'], 
                        marker='s', linewidth=2, color='red')
        axes[0, 1].set_title('Total Fatalities by Year')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Number of Fatalities')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Average fatalities per crash
        axes[1, 0].plot(yearly_stats['YEAR'], yearly_stats['AVG_FATALITIES_PER_CRASH'], 
                        marker='^', linewidth=2, color='orange')
        axes[1, 0].set_title('Average Fatalities per Crash')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Average Fatalities')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Average vehicles involved
        axes[1, 1].plot(yearly_stats['YEAR'], yearly_stats['AVG_VEHICLES'], 
                        marker='d', linewidth=2, color='green')
        axes[1, 1].set_title('Average Vehicles Involved per Crash')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Average Vehicles')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Year-over-year percentage changes
        plt.figure(figsize=(12, 6))
        years = yearly_stats['YEAR'][1:]  # Skip first year (no change)
        crash_changes = yearly_stats['CRASH_CHANGE_PCT'][1:]
        fatality_changes = yearly_stats['FATALITY_CHANGE_PCT'][1:]
        
        x = np.arange(len(years))
        width = 0.35
        
        plt.bar(x - width/2, crash_changes, width, label='Crash Change %', alpha=0.7)
        plt.bar(x + width/2, fatality_changes, width, label='Fatality Change %', alpha=0.7)
        
        plt.xlabel('Year')
        plt.ylabel('Percentage Change')
        plt.title('Year-over-Year Percentage Changes')
        plt.xticks(x, years, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def predict_2025(self):
        """Predict 2025 totals based on current data and historical patterns."""
        if not self.loaded:
            print("Please load data first using load_data()")
            return
            
        print("\n" + "="*60)
        print("2025 PREDICTION ANALYSIS")
        print("="*60)
        
        # Get 2025 data so far
        data_2025 = self.recent_df[self.recent_df['YEAR'] == 2025]
        current_2025_count = len(data_2025)
        
        print(f"\nCurrent 2025 data: {current_2025_count:,} crashes")
        
        # Calculate what day of the year we are
        current_date = datetime.now()
        day_of_year = current_date.timetuple().tm_yday
        days_in_year = 366 if current_date.year % 4 == 0 else 365  # Account for leap year
        progress_through_year = day_of_year / days_in_year
        
        print(f"Current date: {current_date.strftime('%Y-%m-%d')}")
        print(f"Progress through 2025: {progress_through_year:.1%}")
        
        # Predict total for 2025 based on current pace
        predicted_2025_total = int(current_2025_count / progress_through_year)
        
        print(f"\nPredicted 2025 total: {predicted_2025_total:,} crashes")
        
        # Compare with previous years
        yearly_stats = self.recent_df.groupby('YEAR').size()
        recent_years = yearly_stats[yearly_stats.index >= 2020]
        
        print(f"\nComparison with recent years:")
        for year, count in recent_years.items():
            if year < 2025:
                change_from_predicted = ((predicted_2025_total - count) / count) * 100
                print(f"  {year}: {count:,} crashes (2025 prediction: {change_from_predicted:+.1f}%)")
        
        # Seasonal analysis for more accurate prediction
        final_prediction = self._seasonal_prediction_analysis()
        
        # Visualize prediction
        self._plot_2025_prediction(final_prediction, yearly_stats)
        
    def _seasonal_prediction_analysis(self):
        """Analyze seasonal patterns to improve 2025 prediction."""
        print(f"\nSeasonal Analysis for 2025 Prediction:")
        
        # Add month to recent data
        self.recent_df['MONTH'] = self.recent_df['REPORT_DATE'].dt.month
        
        # Calculate monthly averages for recent years (2020-2024) - CORRECTED
        recent_data = self.recent_df[self.recent_df['YEAR'].between(2020, 2024)]
        monthly_totals = recent_data.groupby('MONTH').size()
        monthly_avg_corrected = monthly_totals / 5  # Divide by 5 years to get per-year average
        
        # Get 2025 data by month so far
        data_2025_by_month = self.recent_df[self.recent_df['YEAR'] == 2025].groupby('MONTH').size()
        
        print(f"Monthly averages (2020-2024) - per year:")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in range(1, 13):
            if month in monthly_avg_corrected.index:
                print(f"  {month_names[month-1]}: {monthly_avg_corrected[month]:,.0f} crashes")
        
        print(f"\n2025 data by month so far:")
        for month in range(1, 13):
            if month in data_2025_by_month.index:
                print(f"  {month_names[month-1]}: {data_2025_by_month[month]:,.0f} crashes")
        
        # Calculate remaining months for 2025 with CORRECTED averages
        current_month = datetime.now().month
        remaining_months = 12 - current_month
        
        # Predict remaining crashes based on corrected historical monthly averages
        predicted_remaining = 0
        for month in range(current_month + 1, 13):
            if month in monthly_avg_corrected.index:
                predicted_remaining += monthly_avg_corrected[month]
        
        current_2025_total = len(self.recent_df[self.recent_df['YEAR'] == 2025])
        seasonal_prediction = current_2025_total + predicted_remaining
        
        print(f"\nCurrent 2025 crashes: {current_2025_total:,}")
        print(f"Predicted remaining 2025 crashes: {predicted_remaining:.0f}")
        print(f"Seasonal prediction for 2025 total: {seasonal_prediction:.0f}")
        
        # Alternative prediction based on recent trend
        recent_yearly_avg = self.recent_df[self.recent_df['YEAR'].between(2020, 2024)].groupby('YEAR').size().mean()
        trend_prediction = recent_yearly_avg
        
        print(f"\nRecent yearly average (2020-2024): {recent_yearly_avg:.0f}")
        print(f"Trend-based prediction for 2025: {trend_prediction:.0f}")
        
        # Use the higher of the two predictions
        final_prediction = max(seasonal_prediction, trend_prediction)
        print(f"\nFinal 2025 prediction: {final_prediction:.0f} crashes")
        
        return final_prediction
        
    def _plot_2025_prediction(self, predicted_total, yearly_stats):
        """Create visualization showing 2025 prediction."""
        
        # Prepare data for plotting
        years = list(yearly_stats.index) + [2025]
        counts = list(yearly_stats.values) + [predicted_total]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(years[:-1], counts[:-1], marker='o', linewidth=2, color='blue', label='Historical')
        
        # Plot prediction
        plt.plot(years[-2:], counts[-2:], marker='s', linewidth=2, color='red', linestyle='--', label='2025 Prediction')
        
        # Add annotation for prediction
        plt.annotate(f'Predicted: {predicted_total:,}', 
                    xy=(2025, predicted_total), 
                    xytext=(2024, predicted_total * 1.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        plt.title('Crash Trends with 2025 Prediction')
        plt.xlabel('Year')
        plt.ylabel('Number of Crashes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def monthly_pattern_analysis(self):
        """Analyze monthly patterns across the 10-year period."""
        if not self.loaded:
            print("Please load data first using load_data()")
            return
            
        print("\n" + "="*60)
        print("MONTHLY PATTERN ANALYSIS")
        print("="*60)
        
        # Add month to data
        self.recent_df['MONTH'] = self.recent_df['REPORT_DATE'].dt.month
        
        # Monthly statistics
        monthly_stats = self.recent_df.groupby('MONTH').agg({
            'FATALITIES': ['sum', 'mean'],
            'INJURIES': ['sum', 'mean'],
            'CRASH_ID': 'count'
        }).round(2)
        
        monthly_stats.columns = ['TOTAL_FATALITIES', 'AVG_FATALITIES', 'TOTAL_INJURIES', 'AVG_INJURIES', 'TOTAL_CRASHES']
        monthly_stats = monthly_stats.reset_index()
        
        print("\nMonthly Statistics (2015-2025):")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_stats['MONTH_NAME'] = [month_names[i-1] for i in monthly_stats['MONTH']]
        print(monthly_stats[['MONTH_NAME', 'TOTAL_CRASHES', 'TOTAL_FATALITIES', 'TOTAL_INJURIES']].to_string(index=False))
        
        # Visualize monthly patterns
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Total crashes by month
        plt.subplot(1, 3, 1)
        plt.bar(monthly_stats['MONTH'], monthly_stats['TOTAL_CRASHES'], color='skyblue')
        plt.title('Total Crashes by Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Crashes')
        plt.xticks(range(1, 13), month_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Average fatalities by month
        plt.subplot(1, 3, 2)
        plt.bar(monthly_stats['MONTH'], monthly_stats['AVG_FATALITIES'], color='red', alpha=0.7)
        plt.title('Average Fatalities per Crash by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Fatalities')
        plt.xticks(range(1, 13), month_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Average injuries by month
        plt.subplot(1, 3, 3)
        plt.bar(monthly_stats['MONTH'], monthly_stats['AVG_INJURIES'], color='orange', alpha=0.7)
        plt.title('Average Injuries per Crash by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Injuries')
        plt.xticks(range(1, 13), month_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def summary_insights(self):
        """Generate summary insights for the recent trends."""
        if not self.loaded:
            print("Please load data first using load_data()")
            return
            
        print("\n" + "="*60)
        print("SUMMARY INSIGHTS (2015-2025)")
        print("="*60)
        
        insights = []
        
        # Basic statistics
        insights.append(f"Total crashes in last 10 years: {len(self.recent_df):,}")
        insights.append(f"Average crashes per year: {len(self.recent_df)/10:,.0f}")
        
        # Severity insights
        total_fatalities = self.recent_df['FATALITIES'].sum()
        total_injuries = self.recent_df['INJURIES'].sum()
        insights.append(f"Total fatalities: {total_fatalities:,}")
        insights.append(f"Total injuries: {total_injuries:,}")
        
        # Year range
        year_range = f"{self.recent_df['YEAR'].min()} to {self.recent_df['YEAR'].max()}"
        insights.append(f"Data spans: {year_range}")
        
        # Peak year
        peak_year = self.recent_df.groupby('YEAR').size().idxmax()
        peak_count = self.recent_df.groupby('YEAR').size().max()
        insights.append(f"Peak year: {peak_year} with {peak_count:,} crashes")
        
        # Recent trend
        recent_years = self.recent_df[self.recent_df['YEAR'] >= 2020]
        recent_avg = len(recent_years) / 5
        insights.append(f"Recent average (2020-2024): {recent_avg:,.0f} crashes per year")
        
        print("\nKey Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
    
    def run_full_analysis(self):
        """Run the complete recent trends analysis."""
        print("Starting recent crash trends analysis (2015-2025)...")
        
        if not self.load_data():
            return
            
        self.basic_recent_info()
        self.year_by_year_analysis()
        self.predict_2025()
        self.monthly_pattern_analysis()
        self.summary_insights()


def main():
    """Main function to run the recent trends analysis."""
    # Initialize the analyzer
    data_path = "./data/Crash_File_20250712.csv"
    analyzer = RecentTrendsAnalyzer(data_path)
    
    # Run the full analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main() 