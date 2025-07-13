"""
Kansas City Year-by-Year Crash Analysis (2015-2024)
===================================================

This script provides a comprehensive year-by-year analysis of crash data for Kansas City (MO and KS)
for the last 10 years (2015-2024), excluding 2025 due to partial data. It includes detailed comparisons,
trends, and new visualizations for year-over-year analysis, focusing only on Kansas City records.
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

class KCYearByYearAnalyzer:
    """
    Kansas City-focused year-by-year crash data analyzer for the last 10 years (2015-2024).
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.kc_df = None
        self.loaded = False
        self.years = list(range(2015, 2025))  # 2015-2024
    
    def load_data(self):
        print("Loading crash data...")
        print(f"File path: {self.data_path}")
        try:
            self.df = pd.read_csv(self.data_path, low_memory=False)
            self.loaded = True
            print(f"✓ Data loaded successfully!")
            print(f"Total dataset shape: {self.df.shape}")
            self._process_date_columns()
            self._filter_kansas_city_data()
            self._filter_recent_years()
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
        return True
    
    def _process_date_columns(self):
        print("\nProcessing date columns...")
        if 'REPORT_DATE' in self.df.columns:
            self.df['REPORT_DATE'] = pd.to_datetime(self.df['REPORT_DATE'], format='%Y%m%d', errors='coerce')
        if 'CHANGE_DATE' in self.df.columns:
            self.df['CHANGE_DATE'] = pd.to_datetime(self.df['CHANGE_DATE'], format='%Y%m%d %H%M', errors='coerce')
        if 'REPORT_TIME' in self.df.columns:
            self.df['REPORT_TIME_STR'] = self.df['REPORT_TIME'].astype(str).str.zfill(4)
            self.df['REPORT_HOUR'] = self.df['REPORT_TIME_STR'].str[:2].astype(int, errors='ignore')
        self.df['YEAR'] = self.df['REPORT_DATE'].dt.year
        self.df['MONTH'] = self.df['REPORT_DATE'].dt.month
        self.df['DAY_OF_WEEK'] = self.df['REPORT_DATE'].dt.dayofweek
        self.df['DAY_NAME'] = self.df['REPORT_DATE'].dt.day_name()
        self.df['MONTH_NAME'] = self.df['REPORT_DATE'].dt.month_name()
        self.df['QUARTER'] = self.df['REPORT_DATE'].dt.quarter
        print("✓ Date columns processed")
    
    def _filter_kansas_city_data(self):
        print("\nFiltering for Kansas City data...")
        kc_filters = (
            (self.df['CITY'].str.contains('KANSAS CITY', case=False, na=False)) |
            (self.df['CITY'].str.contains('KC', case=False, na=False)) |
            (self.df['CRASH_CARRIER_CITY'].str.contains('KANSAS CITY', case=False, na=False)) |
            (self.df['CRASH_CARRIER_CITY'].str.contains('KC', case=False, na=False))
        )
        self.kc_df = self.df[kc_filters].copy()
        print(f"✓ Kansas City data filtered: {self.kc_df.shape[0]:,} records")
        if not self.kc_df.empty:
            print(f"Year range: {self.kc_df['YEAR'].min()} to {self.kc_df['YEAR'].max()}")
    
    def _filter_recent_years(self):
        print("\nFiltering for last 10 years (2015-2024)...")
        self.kc_df = self.kc_df[self.kc_df['YEAR'].isin(self.years)].copy()
        print(f"✓ Recent KC data filtered: {self.kc_df.shape[0]:,} records")
        if not self.kc_df.empty:
            print(f"Year range: {self.kc_df['YEAR'].min()} to {self.kc_df['YEAR'].max()}")
            print("\nCrashes by year:")
            yearly_counts = self.kc_df['YEAR'].value_counts().sort_index()
            for year, count in yearly_counts.items():
                print(f"  {year}: {count:,} crashes")
    
    # The rest of the analysis methods are identical to YearByYearAnalyzer,
    # but use self.kc_df instead of self.recent_df or self.df.
    # For brevity, only the data filtering and setup is shown here.
    # You can copy the analysis methods from year_by_year_analysis.py and replace self.recent_df with self.kc_df.
    # ...
    
    def basic_overview(self):
        if not self.loaded or self.kc_df.empty:
            print("No Kansas City data available. Please load data first.")
            return
        print("\n" + "="*70)
        print("KANSAS CITY 10-YEAR CRASH DATA OVERVIEW (2015-2024)")
        print("="*70)
        print(f"Dataset shape: {self.kc_df.shape}")
        print(f"Date range: {self.kc_df['YEAR'].min()} to {self.kc_df['YEAR'].max()}")
        print(f"Total crashes: {len(self.kc_df):,}")
        print(f"Average crashes per year: {len(self.kc_df)/10:,.0f}")
        if 'FATALITIES' in self.kc_df.columns:
            total_fatalities = self.kc_df['FATALITIES'].sum()
            fatal_crashes = (self.kc_df['FATALITIES'] > 0).sum()
            print(f"Total fatalities: {total_fatalities:,}")
            print(f"Fatal crashes: {fatal_crashes:,} ({(fatal_crashes/len(self.kc_df))*100:.1f}%)")
            print(f"Average fatalities per year: {total_fatalities/10:,.0f}")
        if 'INJURIES' in self.kc_df.columns:
            total_injuries = self.kc_df['INJURIES'].sum()
            injury_crashes = (self.kc_df['INJURIES'] > 0).sum()
            print(f"Total injuries: {total_injuries:,}")
            print(f"Injury crashes: {injury_crashes:,} ({(injury_crashes/len(self.kc_df))*100:.1f}%)")
            print(f"Average injuries per year: {total_injuries/10:,.0f}")
        print(f"\nBreakdown by state:")
        state_breakdown = self.kc_df['REPORT_STATE'].value_counts()
        for state, count in state_breakdown.items():
            print(f"  {state}: {count:,} crashes ({(count/len(self.kc_df))*100:.1f}%)")
    # ...
    # Copy all other analysis methods from year_by_year_analysis.py, replacing self.recent_df with self.kc_df
    # ...
    def yearly_trends_analysis(self):
        """Comprehensive yearly trends analysis with year-over-year comparisons."""
        if not self.loaded or self.kc_df.empty:
            return
            
        print("\n" + "="*70)
        print("KANSAS CITY YEARLY TRENDS ANALYSIS (2015-2024)")
        print("="*70)
        
        # Calculate yearly statistics
        yearly_stats = self.kc_df.groupby('YEAR').agg({
            'FATALITIES': ['sum', 'mean', 'count'],
            'INJURIES': ['sum', 'mean'],
            'VEHICLES_IN_ACCIDENT': ['mean'],
            'REPORT_STATE': 'nunique'
        }).round(2)
        
        yearly_stats.columns = ['Total_Fatalities', 'Avg_Fatalities', 'Total_Crashes', 
                              'Total_Injuries', 'Avg_Injuries', 'Avg_Vehicles', 'States_Involved']
        
        print("\nYearly Statistics:")
        print(yearly_stats)
        
        # Calculate year-over-year changes
        print("\nYear-over-Year Changes (%):")
        yoy_changes = yearly_stats.pct_change() * 100
        print(yoy_changes.round(2))
        
        # Create comprehensive yearly visualizations
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Total crashes by year
        yearly_crashes = self.kc_df.groupby('YEAR').size()
        axes[0,0].plot(yearly_crashes.index, yearly_crashes.values, marker='o', linewidth=2, color='blue')
        axes[0,0].set_title('Kansas City Total Crashes by Year')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Number of Crashes')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Fatalities by year
        if 'FATALITIES' in self.kc_df.columns:
            yearly_fatalities = self.kc_df.groupby('YEAR')['FATALITIES'].sum()
            axes[0,1].plot(yearly_fatalities.index, yearly_fatalities.values, marker='s', linewidth=2, color='red')
            axes[0,1].set_title('Kansas City Total Fatalities by Year')
            axes[0,1].set_xlabel('Year')
            axes[0,1].set_ylabel('Number of Fatalities')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Injuries by year
        if 'INJURIES' in self.kc_df.columns:
            yearly_injuries = self.kc_df.groupby('YEAR')['INJURIES'].sum()
            axes[0,2].plot(yearly_injuries.index, yearly_injuries.values, marker='^', linewidth=2, color='orange')
            axes[0,2].set_title('Kansas City Total Injuries by Year')
            axes[0,2].set_xlabel('Year')
            axes[0,2].set_ylabel('Number of Injuries')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. Average fatalities per crash by year
        if 'FATALITIES' in self.kc_df.columns:
            avg_fatalities = self.kc_df.groupby('YEAR')['FATALITIES'].mean()
            axes[1,0].plot(avg_fatalities.index, avg_fatalities.values, marker='o', linewidth=2, color='darkred')
            axes[1,0].set_title('Kansas City Average Fatalities per Crash by Year')
            axes[1,0].set_xlabel('Year')
            axes[1,0].set_ylabel('Average Fatalities')
            axes[1,0].grid(True, alpha=0.3)
        
        # 5. Average injuries per crash by year
        if 'INJURIES' in self.kc_df.columns:
            avg_injuries = self.kc_df.groupby('YEAR')['INJURIES'].mean()
            axes[1,1].plot(avg_injuries.index, avg_injuries.values, marker='s', linewidth=2, color='darkorange')
            axes[1,1].set_title('Kansas City Average Injuries per Crash by Year')
            axes[1,1].set_xlabel('Year')
            axes[1,1].set_ylabel('Average Injuries')
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. States involved by year
        states_by_year = self.kc_df.groupby('YEAR')['REPORT_STATE'].nunique()
        axes[1,2].plot(states_by_year.index, states_by_year.values, marker='^', linewidth=2, color='green')
        axes[1,2].set_title('Kansas City Number of States Involved by Year')
        axes[1,2].set_xlabel('Year')
        axes[1,2].set_ylabel('Number of States')
        axes[1,2].grid(True, alpha=0.3)
        
        # 7. Year-over-year crash change
        yoy_crash_change = yearly_crashes.pct_change() * 100
        axes[2,0].bar(yoy_crash_change.index[1:], yoy_crash_change.values[1:], color='lightblue')
        axes[2,0].set_title('Kansas City Year-over-Year Crash Change (%)')
        axes[2,0].set_xlabel('Year')
        axes[2,0].set_ylabel('Percentage Change')
        axes[2,0].grid(True, alpha=0.3)
        axes[2,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 8. Year-over-year fatality change
        if 'FATALITIES' in self.kc_df.columns:
            yoy_fatality_change = yearly_fatalities.pct_change() * 100
            axes[2,1].bar(yoy_fatality_change.index[1:], yoy_fatality_change.values[1:], color='lightcoral')
            axes[2,1].set_title('Kansas City Year-over-Year Fatality Change (%)')
            axes[2,1].set_xlabel('Year')
            axes[2,1].set_ylabel('Percentage Change')
            axes[2,1].grid(True, alpha=0.3)
            axes[2,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 9. Cumulative crashes over time
        cumulative_crashes = yearly_crashes.cumsum()
        axes[2,2].plot(cumulative_crashes.index, cumulative_crashes.values, marker='o', linewidth=2, color='purple')
        axes[2,2].set_title('Kansas City Cumulative Crashes Over Time')
        axes[2,2].set_xlabel('Year')
        axes[2,2].set_ylabel('Cumulative Crashes')
        axes[2,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def seasonal_patterns_by_year(self):
        """Analyze seasonal patterns and how they change year by year."""
        if not self.loaded or self.kc_df.empty:
            return
            
        print("\n" + "="*70)
        print("KANSAS CITY SEASONAL PATTERNS BY YEAR (2015-2024)")
        print("="*70)
        
        # Monthly patterns by year
        monthly_by_year = self.kc_df.groupby(['YEAR', 'MONTH']).size().unstack(fill_value=0)
        
        print("\nMonthly crash patterns by year:")
        print(monthly_by_year)
        
        # Create seasonal visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Heatmap of crashes by month and year
        sns.heatmap(monthly_by_year, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0,0])
        axes[0,0].set_title('Kansas City Crash Heatmap: Month vs Year')
        axes[0,0].set_xlabel('Month')
        axes[0,0].set_ylabel('Year')
        
        # 2. Monthly averages across all years
        monthly_avg = monthly_by_year.mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[0,1].bar(range(1, 13), monthly_avg, color='skyblue')
        axes[0,1].set_title('Kansas City Average Monthly Crashes (2015-2024)')
        axes[0,1].set_xlabel('Month')
        axes[0,1].set_ylabel('Average Crashes')
        axes[0,1].set_xticks(range(1, 13))
        axes[0,1].set_xticklabels(month_names)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Quarterly patterns by year
        quarterly_by_year = self.kc_df.groupby(['YEAR', 'QUARTER']).size().unstack(fill_value=0)
        quarterly_by_year.plot(kind='bar', ax=axes[1,0], width=0.8)
        axes[1,0].set_title('Kansas City Quarterly Crashes by Year')
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Number of Crashes')
        axes[1,0].legend(['Q1', 'Q2', 'Q3', 'Q4'])
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Seasonal trend lines
        for quarter in [1, 2, 3, 4]:
            quarter_data = quarterly_by_year[quarter]
            axes[1,1].plot(quarter_data.index, quarter_data.values, marker='o', linewidth=2, label=f'Q{quarter}')
        axes[1,1].set_title('Kansas City Quarterly Trends Over Time')
        axes[1,1].set_xlabel('Year')
        axes[1,1].set_ylabel('Number of Crashes')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def severity_trends_by_year(self):
        """Analyze how crash severity patterns change year by year."""
        if not self.loaded or self.kc_df.empty:
            return
            
        print("\n" + "="*70)
        print("KANSAS CITY SEVERITY TRENDS BY YEAR (2015-2024)")
        print("="*70)
        
        # Calculate severity metrics by year
        severity_by_year = self.kc_df.groupby('YEAR').agg({
            'FATALITIES': ['sum', 'mean', 'max'],
            'INJURIES': ['sum', 'mean', 'max'],
            'VEHICLES_IN_ACCIDENT': ['mean', 'max']
        }).round(2)
        
        severity_by_year.columns = ['Total_Fatalities', 'Avg_Fatalities', 'Max_Fatalities',
                                  'Total_Injuries', 'Avg_Injuries', 'Max_Injuries',
                                  'Avg_Vehicles', 'Max_Vehicles']
        
        print("\nSeverity Statistics by Year:")
        print(severity_by_year)
        
        # Create severity visualizations
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Fatal crash rate by year
        fatal_rate_by_year = (self.kc_df.groupby('YEAR')['FATALITIES'].apply(lambda x: (x > 0).sum()) / 
                             self.kc_df.groupby('YEAR').size()) * 100
        axes[0,0].plot(fatal_rate_by_year.index, fatal_rate_by_year.values, marker='o', linewidth=2, color='red')
        axes[0,0].set_title('Kansas City Fatal Crash Rate by Year (%)')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Percentage of Fatal Crashes')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Injury crash rate by year
        injury_rate_by_year = (self.kc_df.groupby('YEAR')['INJURIES'].apply(lambda x: (x > 0).sum()) / 
                              self.kc_df.groupby('YEAR').size()) * 100
        axes[0,1].plot(injury_rate_by_year.index, injury_rate_by_year.values, marker='s', linewidth=2, color='orange')
        axes[0,1].set_title('Kansas City Injury Crash Rate by Year (%)')
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Percentage of Injury Crashes')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Average fatalities per fatal crash
        avg_fatalities_per_fatal = (self.kc_df.groupby('YEAR')['FATALITIES'].sum() / 
                                   self.kc_df.groupby('YEAR')['FATALITIES'].apply(lambda x: (x > 0).sum()))
        axes[0,2].plot(avg_fatalities_per_fatal.index, avg_fatalities_per_fatal.values, marker='^', linewidth=2, color='darkred')
        axes[0,2].set_title('Kansas City Avg Fatalities per Fatal Crash')
        axes[0,2].set_xlabel('Year')
        axes[0,2].set_ylabel('Average Fatalities')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Distribution of fatalities by year (box plot)
        fatality_data = [self.kc_df[self.kc_df['YEAR'] == year]['FATALITIES'].values 
                        for year in self.years]
        axes[1,0].boxplot(fatality_data, labels=self.years)
        axes[1,0].set_title('Kansas City Fatality Distribution by Year')
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Number of Fatalities')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Distribution of injuries by year (box plot)
        injury_data = [self.kc_df[self.kc_df['YEAR'] == year]['INJURIES'].values 
                      for year in self.years]
        axes[1,1].boxplot(injury_data, labels=self.years)
        axes[1,1].set_title('Kansas City Injury Distribution by Year')
        axes[1,1].set_xlabel('Year')
        axes[1,1].set_ylabel('Number of Injuries')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Vehicles involved by year
        vehicles_by_year = self.kc_df.groupby('YEAR')['VEHICLES_IN_ACCIDENT'].mean()
        axes[1,2].plot(vehicles_by_year.index, vehicles_by_year.values, marker='o', linewidth=2, color='blue')
        axes[1,2].set_title('Kansas City Average Vehicles Involved by Year')
        axes[1,2].set_xlabel('Year')
        axes[1,2].set_ylabel('Average Vehicles')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def state_comparison_by_year(self):
        """Compare how different states perform year by year."""
        if not self.loaded or self.kc_df.empty:
            return
            
        print("\n" + "="*70)
        print("KANSAS CITY STATE COMPARISON BY YEAR (2015-2024)")
        print("="*70)
        
        # Top states by total crashes
        top_states = self.kc_df['REPORT_STATE'].value_counts().head(5).index
        
        print(f"\nAnalyzing top states: {list(top_states)}")
        
        # Create state comparison visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Top states crash trends over time
        state_trends = self.kc_df[self.kc_df['REPORT_STATE'].isin(top_states)].groupby(['YEAR', 'REPORT_STATE']).size().unstack(fill_value=0)
        state_trends.plot(kind='line', marker='o', ax=axes[0,0], linewidth=2)
        axes[0,0].set_title('Kansas City Top States: Crash Trends Over Time')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Number of Crashes')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. State ranking changes over time
        state_rankings = self.kc_df.groupby(['YEAR', 'REPORT_STATE']).size().unstack(fill_value=0)
        top_3_rankings = state_rankings.rank(axis=1, ascending=False)[top_states[:3]]
        top_3_rankings.plot(kind='line', marker='s', ax=axes[0,1], linewidth=2)
        axes[0,1].set_title('Kansas City Top 3 States: Ranking Changes Over Time')
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Rank (1 = Highest)')
        axes[0,1].invert_yaxis()
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. State fatality rates by year
        if 'FATALITIES' in self.kc_df.columns:
            state_fatality_rates = (self.kc_df.groupby(['YEAR', 'REPORT_STATE'])['FATALITIES'].apply(lambda x: (x > 0).sum()) / 
                                  self.kc_df.groupby(['YEAR', 'REPORT_STATE']).size()) * 100
            state_fatality_rates = state_fatality_rates.unstack(fill_value=0)
            top_3_fatality = state_fatality_rates[top_states[:3]]
            top_3_fatality.plot(kind='line', marker='^', ax=axes[1,0], linewidth=2)
            axes[1,0].set_title('Kansas City Top 3 States: Fatality Rates Over Time')
            axes[1,0].set_xlabel('Year')
            axes[1,0].set_ylabel('Fatality Rate (%)')
            axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. State growth rates (comparing 2015 to 2024)
        state_growth = (state_trends.iloc[-1] / state_trends.iloc[0] - 1) * 100
        state_growth = state_growth.sort_values(ascending=False)
        axes[1,1].bar(range(len(state_growth)), state_growth.values, color='lightgreen')
        axes[1,1].set_title('Kansas City State Growth Rates (2015-2024)')
        axes[1,1].set_xlabel('States')
        axes[1,1].set_ylabel('Growth Rate (%)')
        axes[1,1].set_xticks(range(len(state_growth)))
        axes[1,1].set_xticklabels(state_growth.index, rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def vehicle_trends_by_year(self):
        """Analyze how vehicle characteristics change year by year."""
        if not self.loaded or self.kc_df.empty:
            return
            
        print("\n" + "="*70)
        print("KANSAS CITY VEHICLE TRENDS BY YEAR (2015-2024)")
        print("="*70)
        
        # Create vehicle trend visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Truck vs Bus distribution by year
        if 'TRUCK_BUS_IND' in self.kc_df.columns:
            truck_bus_by_year = self.kc_df.groupby(['YEAR', 'TRUCK_BUS_IND']).size().unstack(fill_value=0)
            truck_bus_by_year.plot(kind='bar', ax=axes[0,0], width=0.8)
            axes[0,0].set_title('Kansas City Truck vs Bus Crashes by Year')
            axes[0,0].set_xlabel('Year')
            axes[0,0].set_ylabel('Number of Crashes')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Vehicle configuration trends
        if 'VEHICLE_CONFIGURATION_ID' in self.kc_df.columns:
            top_configs = self.kc_df['VEHICLE_CONFIGURATION_ID'].value_counts().head(5).index
            config_by_year = self.kc_df[self.kc_df['VEHICLE_CONFIGURATION_ID'].isin(top_configs)].groupby(['YEAR', 'VEHICLE_CONFIGURATION_ID']).size().unstack(fill_value=0)
            config_by_year.plot(kind='line', marker='o', ax=axes[0,1], linewidth=2)
            axes[0,1].set_title('Kansas City Top 5 Vehicle Configurations Over Time')
            axes[0,1].set_xlabel('Year')
            axes[0,1].set_ylabel('Number of Crashes')
            axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. GVW rating trends
        if 'GVW_RATING_ID' in self.kc_df.columns:
            top_gvw = self.kc_df['GVW_RATING_ID'].value_counts().head(5).index
            gvw_by_year = self.kc_df[self.kc_df['GVW_RATING_ID'].isin(top_gvw)].groupby(['YEAR', 'GVW_RATING_ID']).size().unstack(fill_value=0)
            gvw_by_year.plot(kind='line', marker='s', ax=axes[1,0], linewidth=2)
            axes[1,0].set_title('Kansas City Top 5 GVW Ratings Over Time')
            axes[1,0].set_xlabel('Year')
            axes[1,0].set_ylabel('Number of Crashes')
            axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Vehicle configuration vs severity over time
        if 'VEHICLE_CONFIGURATION_ID' in self.kc_df.columns and 'FATALITIES' in self.kc_df.columns:
            config_severity_by_year = self.kc_df.groupby(['YEAR', 'VEHICLE_CONFIGURATION_ID'])['FATALITIES'].mean().unstack(fill_value=0)
            top_severe_configs = config_severity_by_year.mean().sort_values(ascending=False).head(5).index
            config_severity_by_year[top_severe_configs].plot(kind='line', marker='^', ax=axes[1,1], linewidth=2)
            axes[1,1].set_title('Kansas City Top 5 Most Severe Vehicle Configs Over Time')
            axes[1,1].set_xlabel('Year')
            axes[1,1].set_ylabel('Average Fatalities per Crash')
            axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def environmental_trends_by_year(self):
        """Analyze how environmental factors change year by year."""
        if not self.loaded or self.kc_df.empty:
            return
            
        print("\n" + "="*70)
        print("KANSAS CITY ENVIRONMENTAL TRENDS BY YEAR (2015-2024)")
        print("="*70)
        
        # Create environmental trend visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Weather condition trends
        if 'WEATHER_CONDITION_ID' in self.kc_df.columns:
            top_weather = self.kc_df['WEATHER_CONDITION_ID'].value_counts().head(5).index
            weather_by_year = self.kc_df[self.kc_df['WEATHER_CONDITION_ID'].isin(top_weather)].groupby(['YEAR', 'WEATHER_CONDITION_ID']).size().unstack(fill_value=0)
            weather_by_year.plot(kind='line', marker='o', ax=axes[0,0], linewidth=2)
            axes[0,0].set_title('Kansas City Top 5 Weather Conditions Over Time')
            axes[0,0].set_xlabel('Year')
            axes[0,0].set_ylabel('Number of Crashes')
            axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Light condition trends
        if 'LIGHT_CONDITION_ID' in self.kc_df.columns:
            top_light = self.kc_df['LIGHT_CONDITION_ID'].value_counts().head(5).index
            light_by_year = self.kc_df[self.kc_df['LIGHT_CONDITION_ID'].isin(top_light)].groupby(['YEAR', 'LIGHT_CONDITION_ID']).size().unstack(fill_value=0)
            light_by_year.plot(kind='line', marker='s', ax=axes[0,1], linewidth=2)
            axes[0,1].set_title('Kansas City Top 5 Light Conditions Over Time')
            axes[0,1].set_xlabel('Year')
            axes[0,1].set_ylabel('Number of Crashes')
            axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Road surface condition trends
        if 'ROAD_SURFACE_CONDITION_ID' in self.kc_df.columns:
            top_road = self.kc_df['ROAD_SURFACE_CONDITION_ID'].value_counts().head(5).index
            road_by_year = self.kc_df[self.kc_df['ROAD_SURFACE_CONDITION_ID'].isin(top_road)].groupby(['YEAR', 'ROAD_SURFACE_CONDITION_ID']).size().unstack(fill_value=0)
            road_by_year.plot(kind='line', marker='^', ax=axes[1,0], linewidth=2)
            axes[1,0].set_title('Kansas City Top 5 Road Surface Conditions Over Time')
            axes[1,0].set_xlabel('Year')
            axes[1,0].set_ylabel('Number of Crashes')
            axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Environmental severity correlation over time
        if all(col in self.kc_df.columns for col in ['WEATHER_CONDITION_ID', 'FATALITIES']):
            weather_severity_by_year = self.kc_df.groupby(['YEAR', 'WEATHER_CONDITION_ID'])['FATALITIES'].mean().unstack(fill_value=0)
            top_severe_weather = weather_severity_by_year.mean().sort_values(ascending=False).head(5).index
            weather_severity_by_year[top_severe_weather].plot(kind='line', marker='o', ax=axes[1,1], linewidth=2)
            axes[1,1].set_title('Kansas City Most Severe Weather Conditions Over Time')
            axes[1,1].set_xlabel('Year')
            axes[1,1].set_ylabel('Average Fatalities per Crash')
            axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def comprehensive_summary(self):
        """Generate comprehensive summary and insights for the 10-year period."""
        if not self.loaded or self.kc_df.empty:
            return
            
        print("\n" + "="*70)
        print("KANSAS CITY COMPREHENSIVE 10-YEAR SUMMARY (2015-2024)")
        print("="*70)
        
        insights = []
        
        # Basic statistics
        insights.append(f"Total Kansas City crashes analyzed: {len(self.kc_df):,}")
        insights.append(f"Average Kansas City crashes per year: {len(self.kc_df)/10:,.0f}")
        
        # Severity insights
        if 'FATALITIES' in self.kc_df.columns:
            total_fatalities = self.kc_df['FATALITIES'].sum()
            fatal_crashes = (self.kc_df['FATALITIES'] > 0).sum()
            insights.append(f"Total Kansas City fatalities: {total_fatalities:,}")
            insights.append(f"Average Kansas City fatalities per year: {total_fatalities/10:,.0f}")
            insights.append(f"Kansas City fatal crash rate: {(fatal_crashes/len(self.kc_df))*100:.2f}%")
            
        if 'INJURIES' in self.kc_df.columns:
            total_injuries = self.kc_df['INJURIES'].sum()
            injury_crashes = (self.kc_df['INJURIES'] > 0).sum()
            insights.append(f"Total Kansas City injuries: {total_injuries:,}")
            insights.append(f"Average Kansas City injuries per year: {total_injuries/10:,.0f}")
            insights.append(f"Kansas City injury crash rate: {(injury_crashes/len(self.kc_df))*100:.2f}%")
        
        # Temporal insights
        yearly_crashes = self.kc_df.groupby('YEAR').size()
        peak_year = yearly_crashes.idxmax()
        peak_count = yearly_crashes.max()
        lowest_year = yearly_crashes.idxmin()
        lowest_count = yearly_crashes.min()
        
        insights.append(f"Peak Kansas City crash year: {peak_year} with {peak_count:,} crashes")
        insights.append(f"Lowest Kansas City crash year: {lowest_year} with {lowest_count:,} crashes")
        insights.append(f"Kansas City year-over-year variation: {((peak_count - lowest_count) / lowest_count) * 100:.1f}%")
        
        # Trend analysis
        first_half_avg = yearly_crashes[2015:2020].mean()
        second_half_avg = yearly_crashes[2020:2025].mean()
        trend_direction = "increased" if second_half_avg > first_half_avg else "decreased"
        trend_percentage = abs((second_half_avg - first_half_avg) / first_half_avg) * 100
        insights.append(f"Kansas City overall trend: Crashes {trend_direction} by {trend_percentage:.1f}% from first half to second half")
        
        # State insights
        top_state = self.kc_df['REPORT_STATE'].value_counts().index[0]
        top_state_count = self.kc_df['REPORT_STATE'].value_counts().iloc[0]
        insights.append(f"Kansas City state with most crashes: {top_state} with {top_state_count:,} crashes")
        
        # Vehicle insights
        if 'TRUCK_BUS_IND' in self.kc_df.columns:
            truck_pct = (self.kc_df['TRUCK_BUS_IND'] == 'T').sum() / len(self.kc_df) * 100
            insights.append(f"Kansas City truck crashes: {truck_pct:.1f}% of total")
        
        print("\nKey Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"{i:2d}. {insight}")
        
        print("\n" + "="*70)
        print("KANSAS CITY 10-YEAR ANALYSIS COMPLETE")
        print("="*70)
    def run_full_analysis(self):
        print("Starting comprehensive Kansas City 10-year crash analysis (2015-2024)...")
        if not self.load_data():
            return
        if self.kc_df.empty:
            print("No Kansas City data found in the dataset for 2015-2024.")
            return
        self.basic_overview()
        self.yearly_trends_analysis()
        self.seasonal_patterns_by_year()
        self.severity_trends_by_year()
        self.state_comparison_by_year()
        self.vehicle_trends_by_year()
        self.environmental_trends_by_year()
        self.comprehensive_summary()

def main():
    data_path = "./data/Crash_File_20250712.csv"
    analyzer = KCYearByYearAnalyzer(data_path)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main() 