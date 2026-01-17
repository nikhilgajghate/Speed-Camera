"""
Speed Camera Analysis Pipeline

This module analyzes the relationship between speed cameras and traffic crashes
using a quasi-experimental design approach.
"""

from sklearn.model_selection import train_test_split
from shapely.ops import unary_union
from shapely.geometry import Point
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
import geopandas as gpd
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

# File paths
DATA_DIR = Path('Data')
CAMERA_LOCATIONS_FILE = DATA_DIR / 'Speed_Camera_Locations_20250911.csv'
CAMERA_VIOLATIONS_FILE = DATA_DIR / 'Speed_Camera_Violations.csv'
CRASHES_FILE = DATA_DIR / 'Traffic_Crashes_-_Crashes_20250911.csv'
PEOPLE_FILE = DATA_DIR / 'Traffic_Crashes_-_People_20250911.csv'
VEHICLES_FILE = DATA_DIR / 'Traffic_Crashes_-_Vehicles_20250911.csv'

# Analysis parameters
CAMERA_BUFFER_METERS = 250
RANDOM_SEED = 123
TEST_SIZE = 0.2

# Coordinate reference systems
CRS_WGS84 = 'EPSG:4326'
CRS_ILLINOIS = 'EPSG:3435'  # Illinois State Plane East (meters)


def load_datasets():
    """Load all datasets from CSV files."""
    datasets = {
        'speed_camera_locations': pd.read_csv(CAMERA_LOCATIONS_FILE),
        'speed_camera_violations': pd.read_csv(CAMERA_VIOLATIONS_FILE),
        'crashes': pd.read_csv(CRASHES_FILE),
        'people': pd.read_csv(PEOPLE_FILE),
        'vehicles': pd.read_csv(VEHICLES_FILE),
    }
    
    for name, df in datasets.items():
        print(f"{name.replace('_', ' ').title()} shape: {df.shape}")
    
    return datasets


def clean_column_names(df):
    """Convert column names to snake_case."""
    df.columns = (df.columns
                  .str.lower()
                  .str.replace('.', '_', regex=False)
                  .str.replace(' ', '_', regex=False)
                  .str.replace('__', '_', regex=False))
    return df


def drop_redundant_columns(datasets):
    """Remove columns that are not needed for analysis."""
    if 'second_approach' in datasets['speed_camera_locations'].columns:
        datasets['speed_camera_locations'] = datasets['speed_camera_locations'].drop(
            columns=['second_approach']
        )
    
    people_cols_to_drop = [
        'seat_no', 'drivers_license_class', 'ems_agency', 
        'hospital', 'ems_run_no', 'bac_result_value'
    ]
    existing_cols = [c for c in people_cols_to_drop if c in datasets['people'].columns]
    datasets['people'] = datasets['people'].drop(columns=existing_cols)
    
    vehicles_cols_to_drop = [
        'trailer1_length', 'trailer2_length', 'total_vehicle_length', 
        'ccmc_no', 'ilcc_no', 'idot_permit_no', 'towed_by', 'towed_to', 
        'num_passengers', 'vehicle_year'
    ]
    existing_cols = [c for c in vehicles_cols_to_drop if c in datasets['vehicles'].columns]
    datasets['vehicles'] = datasets['vehicles'].drop(columns=existing_cols)
    
    return datasets


def handle_missing_values(datasets):
    """Drop rows with missing values in critical columns."""
    # Speed camera violations
    print(f"Speed Camera Violations shape before: {datasets['speed_camera_violations'].shape}")
    datasets['speed_camera_violations'] = datasets['speed_camera_violations'].dropna(
        subset=['x_coordinate', 'y_coordinate', 'latitude', 'longitude']
    )
    print(f"Speed Camera Violations shape after: {datasets['speed_camera_violations'].shape}")
    
    # Crashes dataset
    injury_cols = [
        'injuries_total', 'injuries_fatal', 'injuries_incapacitating',
        'injuries_non_incapacitating', 'injuries_reported_not_evident',
        'injuries_no_indication', 'injuries_unknown', 'latitude', 'longitude'
    ]
    existing_injury_cols = [c for c in injury_cols if c in datasets['crashes'].columns]
    print(f"Crashes Dataset shape before: {datasets['crashes'].shape}")
    datasets['crashes'] = datasets['crashes'].dropna(subset=existing_injury_cols)
    print(f"Crashes Dataset shape after: {datasets['crashes'].shape}")
    
    # Vehicles dataset
    print(f"Vehicles Dataset shape before: {datasets['vehicles'].shape}")
    datasets['vehicles'] = datasets['vehicles'].dropna(subset=['vehicle_id'])
    print(f"Vehicles Dataset shape after: {datasets['vehicles'].shape}")
    
    return datasets


def convert_date_columns(datasets):
    """Convert date columns to proper datetime format."""
    # Crashes dataset
    datasets['crashes']['crash_datetime'] = pd.to_datetime(
        datasets['crashes']['crash_date'], format='mixed'
    )
    datasets['crashes']['crash_date'] = pd.to_datetime(
        datasets['crashes']['crash_datetime'].dt.date
    )
    datasets['crashes']['year_month'] = (
        datasets['crashes']['crash_datetime'].dt.to_period('M').dt.to_timestamp()
    )
    
    # Speed camera locations
    datasets['speed_camera_locations']['go_live_date'] = pd.to_datetime(
        datasets['speed_camera_locations']['go_live_date'], format='mixed'
    )
    
    # Speed camera violations
    datasets['speed_camera_violations'] = datasets['speed_camera_violations'].rename(
        columns={'camera_id': 'camera_location_id'}
    )
    datasets['speed_camera_violations']['violation_date'] = pd.to_datetime(
        datasets['speed_camera_violations']['violation_date'], format='mixed'
    )
    datasets['speed_camera_violations']['year_month'] = (
        datasets['speed_camera_violations']['violation_date'].dt.to_period('M').dt.to_timestamp()
    )
    
    return datasets


def convert_violations_to_numeric(violations_df):
    """Convert violations column to numeric and filter NAs."""
    violations_df['violations'] = pd.to_numeric(violations_df['violations'], errors='coerce')
    print(f"NAs in violations: {violations_df['violations'].isna().sum()}")
    
    print(f"Speed Camera Violations shape before filtering NAs: {violations_df.shape}")
    violations_df = violations_df.dropna(subset=['violations'])
    print(f"Speed Camera Violations shape after filtering NAs: {violations_df.shape}")
    
    return violations_df


def add_people_and_vehicle_counts(crashes_df, people_df, vehicles_df):
    """Add counts of people and vehicles involved in each crash."""
    people_counts = people_df.groupby('crash_record_id').size().reset_index(name='num_people')
    vehicles_counts = vehicles_df.groupby('crash_record_id').size().reset_index(name='num_vehicles')
    crashes_df = crashes_df.merge(people_counts, on='crash_record_id', how='left')
    crashes_df = crashes_df.merge(vehicles_counts, on='crash_record_id', how='left')
    
    return crashes_df


def create_geodataframes(crashes_df, camera_locations_df):
    """Convert dataframes to GeoDataFrames with proper CRS."""
    # Crashes GeoDataFrame
    crash_geometry = [
        Point(xy) for xy in zip(crashes_df['longitude'], crashes_df['latitude'])
    ]
    gdf_crashes = gpd.GeoDataFrame(crashes_df, geometry=crash_geometry, crs=CRS_WGS84)
    
    # Camera locations GeoDataFrame
    camera_geometry = [
        Point(xy) for xy in zip(camera_locations_df['longitude'], camera_locations_df['latitude'])
    ]
    gdf_cameras = gpd.GeoDataFrame(camera_locations_df, geometry=camera_geometry, crs=CRS_WGS84)
    
    return gdf_crashes, gdf_cameras


def convert_to_meters(gdf_crashes, gdf_cameras):
    """Convert GeoDataFrames to meters (Illinois State Plane)."""
    return (
        gdf_crashes.to_crs(epsg=3435),
        gdf_cameras.to_crs(epsg=3435)
    )


def find_nearest_cameras(gdf_crashes, gdf_cameras):
    """Find the nearest camera for each crash using KDTree."""
    camera_coords = np.array([
        (geom.x, geom.y) for geom in gdf_cameras.geometry
    ])
    crash_coords = np.array([
        (geom.x, geom.y) for geom in gdf_crashes.geometry
    ])
    
    tree = cKDTree(camera_coords)
    distances, nearest_indices = tree.query(crash_coords)
    
    gdf_crashes['nearest_camera_location_id'] = gdf_cameras.iloc[nearest_indices]['location_id'].values
    gdf_crashes['distance_to_nearest_camera'] = distances
    
    return gdf_crashes


def add_camera_activity_flags(gdf_crashes, camera_locations_df):
    """Add flags indicating if camera was active at time of crash."""
    camera_dates = camera_locations_df[['location_id', 'go_live_date']].rename(
        columns={'location_id': 'nearest_camera_location_id'}
    )
    gdf_crashes = gdf_crashes.merge(camera_dates, on='nearest_camera_location_id', how='left')
    
    gdf_crashes['is_crash_camera_active'] = np.where(
        gdf_crashes['go_live_date'].notna() & 
        (gdf_crashes['crash_date'] >= gdf_crashes['go_live_date']),
        1, 0
    )
    
    return gdf_crashes


def add_proximity_flags(gdf_crashes, gdf_cameras, buffer_meters=CAMERA_BUFFER_METERS):
    """Add flags for crashes within buffer distance of cameras."""
    camera_buffer = gdf_cameras.buffer(buffer_meters)
    buffer_union = unary_union(camera_buffer)
    
    gdf_crashes['is_within_250_meters'] = gdf_crashes.geometry.within(buffer_union).astype(int)
    gdf_crashes['outside_250_meters'] = (~gdf_crashes.geometry.within(buffer_union)).astype(int)
    
    return gdf_crashes


def create_monthly_violations(violations_df):
    """Aggregate violations by camera and month."""
    return violations_df.groupby(['camera_location_id', 'year_month']).agg(
        violations_by_month=('violations', 'sum')
    ).reset_index()


def create_experimental_group(gdf_crashes, camera_locations_df, monthly_violations, is_treatment=True):
    """Create control or treatment group from crash data."""
    group_df = gdf_crashes.drop(columns='geometry').copy()
    
    if is_treatment:
        group_df = group_df[group_df['is_within_250_meters'] == 1]
    else:
        group_df = group_df[group_df['outside_250_meters'] == 1]
    
    # Aggregate by camera and month
    group_df = group_df.groupby(['nearest_camera_location_id', 'year_month']).agg(
        total_crash_count=('crash_record_id', 'count'),
        total_injuries=('injuries_total', lambda x: x.fillna(0).sum()),
        total_fatalities=('injuries_fatal', lambda x: x.fillna(0).sum()),
        posted_speed_limit_median=('posted_speed_limit', 'median')
    ).reset_index()
    
    group_df = group_df.rename(columns={'nearest_camera_location_id': 'camera_location_id'})
    
    # Add go_live_date
    camera_dates = camera_locations_df[['location_id', 'go_live_date']].rename(
        columns={'location_id': 'camera_location_id'}
    )
    group_df = group_df.merge(camera_dates, on='camera_location_id', how='left')
    
    # Convert dates
    group_df['go_live_date'] = pd.to_datetime(group_df['go_live_date'])
    group_df['year_month'] = pd.to_datetime(group_df['year_month'])
    
    # Add camera activity flags
    group_df['is_camera_active'] = np.where(
        group_df['go_live_date'].notna() & 
        (group_df['year_month'] >= group_df['go_live_date'].dt.to_period('M').dt.to_timestamp()),
        1, 0
    )
    
    group_df['months_since_camera_active'] = np.where(
        group_df['go_live_date'].notna(),
        ((group_df['year_month'].dt.year - group_df['go_live_date'].dt.year) * 12 +
         (group_df['year_month'].dt.month - group_df['go_live_date'].dt.month)),
        np.nan
    )
    
    # Add violations
    group_df = group_df.merge(monthly_violations, on=['camera_location_id', 'year_month'], how='left')
    group_df['violations_by_month'] = group_df['violations_by_month'].fillna(0).astype(int)
    group_df['treated'] = 1 if is_treatment else 0
    
    return group_df


def create_panel_data(treatment_df, control_df):
    """Combine treatment and control groups into panel data."""
    panel = pd.concat([treatment_df, control_df], ignore_index=True)
    panel['month'] = panel['year_month'].dt.month
    panel['year'] = panel['year_month'].dt.year
    panel = panel.dropna(subset=['camera_location_id', 'go_live_date', 'months_since_camera_active'])
    return panel


def create_panel_summary(panel_df):
    """Create summary statistics for panel data."""
    summary = panel_df.groupby(['treated', 'year_month']).agg(
        total_crash_count=('total_crash_count', 'sum'),
        total_fatalities=('total_fatalities', 'sum'),
        total_injuries=('total_injuries', 'sum'),
        total_violations=('violations_by_month', 'sum'),
        median_speed_limit=('posted_speed_limit_median', 'median')
    ).reset_index()
    
    summary['group'] = np.where(
        summary['treated'] == 1, 'Treated (<=250m)', 'Control (>500m)'
    )
    
    return summary


def create_pre_post_summary(panel_df):
    """Create pre/post camera activation summary."""
    pre_post = panel_df[panel_df['go_live_date'].notna()].copy()
    
    pre_post['period'] = np.where(
        pre_post['year_month'] < pre_post['go_live_date'].dt.to_period('M').dt.to_timestamp(),
        'Pre go-live date', 'Post go-live date'
    )
    
    pre_post['group'] = np.where(
        pre_post['treated'] == 1, 'Treated (<=250m)', 'Control (>250m)'
    )
    
    # Aggregate
    agg = pre_post.groupby(['group', 'period']).agg(
        avg_crash_count=('total_crash_count', 'mean'),
        avg_monthly_fatalities=('total_fatalities', 'mean'),
        avg_monthly_injuries=('total_injuries', 'mean'),
        total_monthly_violations=('violations_by_month', 'sum'),
        median_speed_limit=('posted_speed_limit_median', 'median')
    ).reset_index()
    
    return agg


def create_avg_crashes_per_camera(panel_df):
    """Calculate average metrics per camera."""
    return panel_df.groupby(['camera_location_id', 'treated']).agg(
        avg_monthly_crashes=('total_crash_count', 'mean'),
        avg_monthly_fatalities=('total_fatalities', 'mean'),
        avg_monthly_injuries=('total_injuries', 'mean'),
        total_monthly_violations=('violations_by_month', 'sum'),
        median_speed_limit=('posted_speed_limit_median', 'median'),
        n_months=('total_crash_count', 'count')
    ).reset_index()


def plot_crash_camera_locations(gdf_crashes, gdf_cameras, output_path):
    """Plot crash locations and camera positions."""
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_crashes.head(10000).plot(ax=ax, color='red', alpha=0.5, markersize=0.6, label='Crashes')
    gdf_cameras.plot(ax=ax, color='blue', markersize=30, label='Speed Cameras')
    ax.legend()
    ax.set_title('Location of Crashes and Speed Cameras')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_crash_distance_distribution(gdf_crashes, output_path, max_distance=3000):
    """Plot histogram of crash distances to nearest camera."""
    crash_by_distance = gdf_crashes[gdf_crashes['distance_to_nearest_camera'] < max_distance]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(crash_by_distance['distance_to_nearest_camera'], bins=20, 
            color='steelblue', edgecolor='white')
    ax.set_xlabel('Distance to nearest speed camera (meters)')
    ax.set_ylabel('Count of crashes')
    ax.set_title('Distribution of Crash Distances to Nearest Camera')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_violations_by_month(monthly_violations, output_path):
    """Plot total violations by month."""
    total = monthly_violations.groupby('year_month').agg(
        sum_violations=('violations_by_month', 'sum')
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(total['year_month'], total['sum_violations'], color='steelblue')
    ax.set_xlabel('Month')
    ax.set_ylabel('Violations')
    ax.set_title('Speed Camera Violations by Month')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_crashes_by_proximity(gdf_crashes, output_path):
    """Plot crash counts within vs outside camera buffer."""
    fig, ax = plt.subplots(figsize=(8, 6))
    crash_counts = gdf_crashes.groupby('is_within_250_meters').size()
    colors = ['gray', 'orange']
    ax.bar(crash_counts.index.astype(str), crash_counts.values, color=colors)
    ax.set_xlabel('Within 250m (0 = No, 1 = Yes)')
    ax.set_ylabel('Crash Count')
    ax.set_title('Crashes Within vs Outside 250m of Speed Cameras')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_monthly_crash_counts(panel_summary, output_path):
    """Plot monthly crash counts for treatment and control groups."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for group in panel_summary['group'].unique():
        group_data = panel_summary[panel_summary['group'] == group]
        linestyle = '-' if 'Treated' in group else '--'
        ax.plot(group_data['year_month'], group_data['total_crash_count'],
                linestyle=linestyle, linewidth=1, label=group)
    
    # Add trendlines
    for group in panel_summary['group'].unique():
        group_data = panel_summary[panel_summary['group'] == group].sort_values('year_month')
        x_numeric = np.arange(len(group_data))
        z = np.polyfit(x_numeric, group_data['total_crash_count'], 3)
        p = np.poly1d(z)
        ax.plot(group_data['year_month'], p(x_numeric),
                linestyle=':', color='blue', linewidth=0.7, alpha=0.7)
    
    # Reference line
    ax.axvline(x=pd.Timestamp('2015-07-01'), color='red', linestyle='--', linewidth=0.8)
    ax.text(pd.Timestamp('2015-07-01'), ax.get_ylim()[1] * 0.5,
            'More consistent data post 07-01-2015', color='red',
            rotation=90, va='bottom', ha='right', fontsize=9)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Crash Count')
    ax.set_title('Monthly Crash Counts: Treated vs Control Areas\n\n'
                 'Control: Crashes reported >500m of speed cameras\n'
                 'Treatment: Crashes reported <=250m of speed cameras')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_monthly_fatality_counts(panel_summary, output_path):
    """Plot monthly fatality counts for treatment and control groups."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for group in panel_summary['group'].unique():
        group_data = panel_summary[panel_summary['group'] == group]
        linestyle = '-' if 'Treated' in group else '--'
        ax.plot(group_data['year_month'], group_data['total_fatalities'],
                linestyle=linestyle, linewidth=1, label=group)
    
    ax.axvline(x=pd.Timestamp('2015-07-01'), color='red', linestyle='--', linewidth=0.8)
    ax.text(pd.Timestamp('2015-07-01'), ax.get_ylim()[1] * 0.5,
            'More consistent data post 07-01-2015', color='red',
            rotation=90, va='bottom', ha='right', fontsize=9)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Fatalities Count')
    ax.set_title('Monthly Fatality Counts: Treated vs Control Areas')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_avg_crash_pre_post(pre_post_summary, output_path):
    """Plot average crash count pre vs post camera go-live."""
    control_data = pre_post_summary[pre_post_summary['group'] == 'Control (>250m)']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(control_data['period'].tolist(), control_data['avg_crash_count'].tolist(), 
           color='steelblue')
    ax.set_xlabel('Period')
    ax.set_ylabel('Average Crash Count')
    ax.set_title('Average Monthly Crash Count: Pre vs Post Camera Go-Live')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def perform_ttest(data, column, group_col='treated'):
    """Perform two-sided t-test between groups."""
    group0 = data[data[group_col] == 0][column]
    group1 = data[data[group_col] == 1][column]
    return stats.ttest_ind(group0, group1, equal_var=False)


def run_ttests(panel_df, avg_per_camera_df):
    """Run all t-tests and print results."""
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS - T-TESTS")
    print("=" * 80)
    
    # Raw data t-tests
    print("\n--- T-Tests on Raw Data ---")
    
    raw_tests = [
        ('total_crash_count', 'Total Crash Count'),
        ('total_fatalities', 'Total Fatalities'),
        ('total_injuries', 'Total Injuries'),
        ('violations_by_month', 'Violations by Month'),
    ]
    
    for col, name in raw_tests:
        result = perform_ttest(panel_df, col)
        print(f"\nT-Test: {name}")
        print(f"  t-statistic: {result.statistic:.4f}")
        print(f"  p-value: {result.pvalue:.6f}")
    
    # Aggregate data t-tests
    print("\n--- T-Tests on Aggregate Data ---")
    
    agg_tests = [
        ('avg_monthly_crashes', 'Average Monthly Crashes'),
        ('avg_monthly_fatalities', 'Average Monthly Fatalities'),
        ('avg_monthly_injuries', 'Average Monthly Injuries'),
        ('total_monthly_violations', 'Total Monthly Violations'),
    ]
    
    for col, name in agg_tests:
        result = perform_ttest(avg_per_camera_df, col)
        print(f"\nT-Test: {name}")
        print(f"  t-statistic: {result.statistic:.4f}")
        print(f"  p-value: {result.pvalue:.6f}")


def calculate_conditional_probabilities(panel_df):
    """Calculate conditional probabilities for fatalities and injuries."""
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING")
    print("=" * 80)
    
    panel = panel_df.copy()
    panel['did_fatalities_occur_this_month'] = panel['total_fatalities'] > 0
    panel['did_injuries_occur_this_month'] = panel['total_injuries'] > 0
    
    # Fatality probability
    fatality_prob = panel.groupby('treated').agg(
        num_camera_months=('treated', 'count'),
        months_with_fatalities=('did_fatalities_occur_this_month', 'sum')
    ).reset_index()
    fatality_prob['probability_fatality_given_treat'] = (
        fatality_prob['months_with_fatalities'] / fatality_prob['num_camera_months']
    )
    print("\nFatality Probability by Groups:")
    print(fatality_prob)
    
    # Injury probability
    injury_prob = panel.groupby('treated').agg(
        num_camera_months=('treated', 'count'),
        months_with_injuries=('did_injuries_occur_this_month', 'sum')
    ).reset_index()
    injury_prob['probability_injuries_given_treat'] = (
        injury_prob['months_with_injuries'] / injury_prob['num_camera_months']
    )
    print("\nInjury Probability by Groups:")
    print(injury_prob)
    
    return fatality_prob, injury_prob


def run_ols_regression(X, y, title):
    """Run OLS regression and print summary."""
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    print(f"\n{title}")
    print("-" * len(title))
    print(model.summary())
    return model


def run_regression_models(panel_df):
    """Run all regression models."""
    print("\n" + "=" * 80)
    print("LINEAR REGRESSION ANALYSIS")
    print("=" * 80)
    
    y = panel_df['total_crash_count']
    
    # Model 1: Only treated
    print("\n--- Model 1: Only Treated Variable ---")
    X1 = panel_df[['treated']]
    model1 = run_ols_regression(X1, y, "Model 1: total_crash_count ~ treated")
    
    # Model 2: Treated + fatalities
    print("\n--- Model 2: Treated + Fatalities ---")
    X2 = panel_df[['treated', 'total_fatalities']]
    model2 = run_ols_regression(X2, y, "Model 2: total_crash_count ~ treated + total_fatalities")
    
    # Model 3: Treated + fatalities + injuries
    print("\n--- Model 3: Treated + Fatalities + Injuries ---")
    X3 = panel_df[['treated', 'total_fatalities', 'total_injuries']]
    model3 = run_ols_regression(X3, y, "Model 3: total_crash_count ~ treated + total_fatalities + total_injuries")
    
    # Model 4: + violations
    print("\n--- Model 4: Treated + Fatalities + Injuries + Violations ---")
    X4 = panel_df[['treated', 'total_fatalities', 'total_injuries', 'violations_by_month']]
    model4 = run_ols_regression(X4, y, "Model 4: total_crash_count ~ treated + total_fatalities + total_injuries + violations_by_month")
    
    # Model 5: Full model with interaction
    print("\n--- Model 5: Full Model ---")
    panel_df = panel_df.copy()
    panel_df['camera_active_months_interaction'] = (
        panel_df['is_camera_active'] * panel_df['months_since_camera_active'].fillna(0)
    )
    
    X5 = panel_df[['treated', 'total_injuries', 'posted_speed_limit_median',
                   'is_camera_active', 'camera_active_months_interaction',
                   'violations_by_month']].fillna(0)
    model5 = run_ols_regression(X5, y, "Model 5: Full Regression Model")
    
    # Crash-Fatality correlation
    print("\n--- Crash-Fatality Correlation ---")
    X_crash_fatal = panel_df[['total_fatalities']]
    model_crash_fatal = run_ols_regression(X_crash_fatal, y, "Model: total_crash_count ~ total_fatalities")
    
    return panel_df, [model1, model2, model3, model4, model5, model_crash_fatal]


def rmse(actual, predicted):
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def train_test_evaluation(panel_df):
    """Perform train/test split evaluation."""
    print("\n" + "=" * 80)
    print("TRAIN/TEST SPLIT EVALUATION")
    print("=" * 80)
    
    np.random.seed(RANDOM_SEED)
    
    features = [
        'treated', 'total_injuries', 'posted_speed_limit_median',
        'is_camera_active', 'camera_active_months_interaction', 'violations_by_month'
    ]
    
    # Prepare data
    model_data = panel_df.dropna(subset=features + ['total_crash_count']).copy()
    X = model_data[features]
    y = model_data['total_crash_count']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    
    model = sm.OLS(y_train, X_train_const).fit()
    print("\nModel Summary:")
    print(model.summary())
    
    # Evaluate
    train_predicted = model.predict(X_train_const)
    test_predicted = model.predict(X_test_const)
    
    train_rmse = rmse(y_train, train_predicted)
    test_rmse = rmse(y_test, test_predicted)
    
    print(f"\n--- Model Performance ---")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    return model, train_rmse, test_rmse


def run_data_pipeline():
    """Run the complete data loading and cleaning pipeline."""
    print("=" * 80)
    print("LOADING AND CLEANING DATA")
    print("=" * 80)
    
    # Load data
    datasets = load_datasets()
    
    # Clean column names
    for key in datasets:
        datasets[key] = clean_column_names(datasets[key])
    
    # Clean data
    datasets = drop_redundant_columns(datasets)
    datasets = handle_missing_values(datasets)
    datasets = convert_date_columns(datasets)
    datasets['speed_camera_violations'] = convert_violations_to_numeric(
        datasets['speed_camera_violations']
    )
    
    # Add counts
    datasets['crashes'] = add_people_and_vehicle_counts(
        datasets['crashes'], datasets['people'], datasets['vehicles']
    )
    
    return datasets


def run_geospatial_pipeline(datasets):
    """Run geospatial processing pipeline."""
    print("\n" + "=" * 80)
    print("GEOSPATIAL PROCESSING")
    print("=" * 80)
    
    # Create GeoDataFrames
    gdf_crashes, gdf_cameras = create_geodataframes(
        datasets['crashes'], datasets['speed_camera_locations']
    )
    
    # Convert to meters
    gdf_crashes_m, gdf_cameras_m = convert_to_meters(gdf_crashes, gdf_cameras)
    
    # Find nearest cameras
    gdf_crashes_m = find_nearest_cameras(gdf_crashes_m, gdf_cameras_m)
    
    # Add flags
    gdf_crashes_m = add_camera_activity_flags(gdf_crashes_m, datasets['speed_camera_locations'])
    gdf_crashes_m = add_proximity_flags(gdf_crashes_m, gdf_cameras_m)
    
    # Create monthly violations
    monthly_violations = create_monthly_violations(datasets['speed_camera_violations'])
    
    return gdf_crashes_m, gdf_cameras_m, monthly_violations


def run_experimental_design_pipeline(gdf_crashes, camera_locations, monthly_violations):
    """Run experimental design pipeline."""
    print("\n" + "=" * 80)
    print("EXPERIMENTAL DESIGN: CONTROL & TREATMENT GROUPS")
    print("=" * 80)
    
    # Create groups
    treatment = create_experimental_group(
        gdf_crashes, camera_locations, monthly_violations, is_treatment=True
    )
    control = create_experimental_group(
        gdf_crashes, camera_locations, monthly_violations, is_treatment=False
    )
    
    # Create panel data
    panel = create_panel_data(treatment, control)
    panel_summary = create_panel_summary(panel)
    pre_post_summary = create_pre_post_summary(panel)
    avg_per_camera = create_avg_crashes_per_camera(panel)
    
    return panel, panel_summary, pre_post_summary, avg_per_camera


def run_visualization_pipeline(gdf_crashes, gdf_cameras, monthly_violations, 
                                panel_summary, pre_post_summary):
    """Generate all visualizations."""
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_crash_camera_locations(gdf_crashes, gdf_cameras, DATA_DIR / 'crash_camera_locations.png')
    plot_crash_distance_distribution(gdf_crashes, DATA_DIR / 'crash_distance_distribution.png')
    plot_violations_by_month(monthly_violations, DATA_DIR / 'violations_by_month.png')
    plot_crashes_by_proximity(gdf_crashes, DATA_DIR / 'crashes_within_outside_250m.png')
    plot_monthly_crash_counts(panel_summary, DATA_DIR / 'monthly_crash_counts.png')
    plot_monthly_fatality_counts(panel_summary, DATA_DIR / 'monthly_fatality_counts.png')
    plot_avg_crash_pre_post(pre_post_summary, DATA_DIR / 'avg_crash_pre_post.png')
    
    print("Plots saved successfully.")


def run_analysis_pipeline(panel, avg_per_camera):
    """Run statistical analysis pipeline."""
    # Conditional probabilities
    calculate_conditional_probabilities(panel)
    
    # T-tests
    run_ttests(panel, avg_per_camera)
    
    # Regression models
    panel, models = run_regression_models(panel)
    
    # Train/test evaluation
    model, train_rmse, test_rmse = train_test_evaluation(panel)
    
    return models, (model, train_rmse, test_rmse)


def main():
    """Main entry point for the analysis pipeline."""
    # Data pipeline
    datasets = run_data_pipeline()
    
    # Geospatial pipeline
    gdf_crashes, gdf_cameras, monthly_violations = run_geospatial_pipeline(datasets)
    
    # Experimental design
    panel, panel_summary, pre_post_summary, avg_per_camera = run_experimental_design_pipeline(
        gdf_crashes, datasets['speed_camera_locations'], monthly_violations
    )
    
    # Visualizations
    run_visualization_pipeline(
        gdf_crashes, gdf_cameras, monthly_violations, panel_summary, pre_post_summary
    )
    
    # Statistical analysis
    run_analysis_pipeline(panel, avg_per_camera)
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nPlots saved to Data/ directory:")
    print("  - crash_camera_locations.png")
    print("  - crash_distance_distribution.png")
    print("  - violations_by_month.png")
    print("  - crashes_within_outside_250m.png")
    print("  - monthly_crash_counts.png")
    print("  - monthly_fatality_counts.png")
    print("  - avg_crash_pre_post.png")


if __name__ == "__main__":
    main()
