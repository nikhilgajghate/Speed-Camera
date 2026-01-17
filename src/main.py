from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from shapely.geometry import Point
import matplotlib.pyplot as plt
import statsmodels.api as sm
import geopandas as gpd
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

################################### Import datasets ###################################

# Read Speed Camera Locations dataset
speed_camera_locations = pd.read_csv('Data/Speed_Camera_Locations_20250911.csv')
print(f"Speed Camera Locations shape: {speed_camera_locations.shape}")

# Read Speed Camera Violations dataset
speed_camera_violations = pd.read_csv('Data/Speed_Camera_Violations.csv')
print(f"Speed Camera Violations shape: {speed_camera_violations.shape}")

# Read Traffic Crashes - Crashes dataset
crashes_dataset = pd.read_csv('Data/Traffic_Crashes_-_Crashes_20250911.csv')
print(f"Crashes Dataset shape: {crashes_dataset.shape}")

# Read Traffic Crashes - People dataset
people_dataset = pd.read_csv('Data/Traffic_Crashes_-_People_20250911.csv')
print(f"People Dataset shape: {people_dataset.shape}")

# Read Traffic Crashes - Vehicles dataset
vehicles_dataset = pd.read_csv('Data/Traffic_Crashes_-_Vehicles_20250911.csv')
print(f"Vehicles Dataset shape: {vehicles_dataset.shape}")

################################### Data Wrangling ###################################

# Convert all columns from UPPER CASE to snake_case
def clean_names(df):
    """Convert column names to snake_case"""
    df.columns = (df.columns
                  .str.lower()
                  .str.replace('.', '_', regex=False)
                  .str.replace(' ', '_', regex=False)
                  .str.replace('__', '_', regex=False))
    return df

speed_camera_locations = clean_names(speed_camera_locations)
speed_camera_violations = clean_names(speed_camera_violations)
crashes_dataset = clean_names(crashes_dataset)
people_dataset = clean_names(people_dataset)
vehicles_dataset = clean_names(vehicles_dataset)

# Remove redundant columns
# speed_camera_locations: drop column second_approach because it is not being used
print(f"Speed Camera Locations NAs before: {speed_camera_locations.isna().sum().sum()}")
if 'second_approach' in speed_camera_locations.columns:
    speed_camera_locations = speed_camera_locations.drop(columns=['second_approach'])
print(f"Speed Camera Locations NAs after: {speed_camera_locations.isna().sum().sum()}")

# people_dataset: drop columns seat_no, drivers_license_class, ems_agency, hospital, ems_run_no, bac_result_value
cols_to_drop_people = ['seat_no', 'drivers_license_class', 'ems_agency', 'hospital', 'ems_run_no', 'bac_result_value']
cols_to_drop_people = [col for col in cols_to_drop_people if col in people_dataset.columns]
people_dataset = people_dataset.drop(columns=cols_to_drop_people)

# vehicles_dataset: drop columns
cols_to_drop_vehicles = ['trailer1_length', 'trailer2_length', 'total_vehicle_length', 'ccmc_no', 
                          'ilcc_no', 'idot_permit_no', 'towed_by', 'towed_to', 'num_passengers', 'vehicle_year']
cols_to_drop_vehicles = [col for col in cols_to_drop_vehicles if col in vehicles_dataset.columns]
vehicles_dataset = vehicles_dataset.drop(columns=cols_to_drop_vehicles)

# Drop NAs from speed_camera_violations
print(f"Speed Camera Violations shape before: {speed_camera_violations.shape}")
speed_camera_violations = speed_camera_violations.dropna(subset=['x_coordinate', 'y_coordinate', 'latitude', 'longitude'])
print(f"Speed Camera Violations shape after: {speed_camera_violations.shape}")

# Drop NAs from crashes_dataset
injury_cols = ['injuries_total', 'injuries_fatal', 'injuries_incapacitating', 
               'injuries_non_incapacitating', 'injuries_reported_not_evident', 
               'injuries_no_indication', 'injuries_unknown', 'latitude', 'longitude']
injury_cols = [col for col in injury_cols if col in crashes_dataset.columns]
print(f"Crashes Dataset shape before: {crashes_dataset.shape}")
crashes_dataset = crashes_dataset.dropna(subset=injury_cols)
print(f"Crashes Dataset shape after: {crashes_dataset.shape}")

# Drop NAs from vehicles_dataset
print(f"Vehicles Dataset shape before: {vehicles_dataset.shape}")
vehicles_dataset = vehicles_dataset.dropna(subset=['vehicle_id'])
print(f"Vehicles Dataset shape after: {vehicles_dataset.shape}")

# Convert date format to preferred type
crashes_dataset['crash_datetime'] = pd.to_datetime(crashes_dataset['crash_date'], format='mixed')
crashes_dataset['crash_date'] = crashes_dataset['crash_datetime'].dt.date
crashes_dataset['crash_date'] = pd.to_datetime(crashes_dataset['crash_date'])
crashes_dataset['year_month'] = crashes_dataset['crash_datetime'].dt.to_period('M').dt.to_timestamp()

# Convert date format for speed_camera_locations
speed_camera_locations['go_live_date'] = pd.to_datetime(speed_camera_locations['go_live_date'], format='mixed')

# Rename camera_id to camera_location_id and convert date format
speed_camera_violations = speed_camera_violations.rename(columns={'camera_id': 'camera_location_id'})
speed_camera_violations['violation_date'] = pd.to_datetime(speed_camera_violations['violation_date'], format='mixed')
speed_camera_violations['year_month'] = speed_camera_violations['violation_date'].dt.to_period('M').dt.to_timestamp()

# Convert the violations column from string to numeric
speed_camera_violations['violations'] = pd.to_numeric(speed_camera_violations['violations'], errors='coerce')
print(f"NAs in violations: {speed_camera_violations['violations'].isna().sum()}")

# Filter out NAs in violations
print(f"Speed Camera Violations shape before filtering NAs: {speed_camera_violations.shape}")
speed_camera_violations = speed_camera_violations.dropna(subset=['violations'])
print(f"Speed Camera Violations shape after filtering NAs: {speed_camera_violations.shape}")

# Left join crashes, vehicles and people dataset on crash_record_id column for feature engineering
people_counts = people_dataset.groupby('crash_record_id').size().reset_index(name='num_people')
vehicles_counts = vehicles_dataset.groupby('crash_record_id').size().reset_index(name='num_vehicles')

crashes_dataset = crashes_dataset.merge(people_counts, on='crash_record_id', how='left')
crashes_dataset = crashes_dataset.merge(vehicles_counts, on='crash_record_id', how='left')

# Convert coordinates columns into GeoDataFrame
geometry = [Point(xy) for xy in zip(crashes_dataset['longitude'], crashes_dataset['latitude'])]
sf_crashes_dataset = gpd.GeoDataFrame(crashes_dataset, geometry=geometry, crs='EPSG:4326')

# Speed camera locations GeoDataFrame
geometry_cameras = [Point(xy) for xy in zip(speed_camera_locations['longitude'], speed_camera_locations['latitude'])]
sf_speed_camera_locations = gpd.GeoDataFrame(speed_camera_locations, geometry=geometry_cameras, crs='EPSG:4326')

# Convert the coordinates to meters (Illinois State Plane East - EPSG:3435)
sf_crashes_dataset_meters = sf_crashes_dataset.to_crs(epsg=3435)
sf_speed_camera_locations_meters = sf_speed_camera_locations.to_crs(epsg=3435)

# Find the index of the nearest camera for every crash
from scipy.spatial import cKDTree

# Create KDTree for camera locations
camera_coords = np.array(list(zip(sf_speed_camera_locations_meters.geometry.x, 
                                   sf_speed_camera_locations_meters.geometry.y)))
crash_coords = np.array(list(zip(sf_crashes_dataset_meters.geometry.x, 
                                  sf_crashes_dataset_meters.geometry.y)))

tree = cKDTree(camera_coords)
distances, nearest_indices = tree.query(crash_coords)

# Get the camera id for the nearest camera and distance
sf_crashes_dataset_meters['nearest_camera_location_id'] = sf_speed_camera_locations_meters.iloc[nearest_indices]['location_id'].values
sf_crashes_dataset_meters['distance_to_nearest_camera'] = distances

# Join crashes to the speed camera locations dataset to get the go_live_date
camera_dates = speed_camera_locations[['location_id', 'go_live_date']].rename(
    columns={'location_id': 'nearest_camera_location_id'})
sf_crashes_dataset_meters = sf_crashes_dataset_meters.merge(camera_dates, on='nearest_camera_location_id', how='left')

# Create is_crash_camera_active column
sf_crashes_dataset_meters['is_crash_camera_active'] = np.where(
    sf_crashes_dataset_meters['go_live_date'].notna() & 
    (sf_crashes_dataset_meters['crash_date'] >= sf_crashes_dataset_meters['go_live_date']),
    1, 0)

# Define thresholds (250m buffer around cameras)
threshold_250 = sf_speed_camera_locations_meters.buffer(250)

# Check if crashes are within 250 meters of any camera
from shapely.ops import unary_union
threshold_250_union = unary_union(threshold_250)

sf_crashes_dataset_meters['is_within_250_meters'] = sf_crashes_dataset_meters.geometry.within(threshold_250_union).astype(int)
sf_crashes_dataset_meters['outside_250_meters'] = (~sf_crashes_dataset_meters.geometry.within(threshold_250_union)).astype(int)

# Create a monthly violations variable
monthly_violations = speed_camera_violations.groupby(['camera_location_id', 'year_month']).agg(
    violations_by_month=('violations', 'sum')
).reset_index()

################################### Data Wrangling Graphs ###################################

# Plot the location of the crash and the cameras
fig, ax = plt.subplots(figsize=(12, 10))
sf_crashes_dataset_meters.head(10000).plot(ax=ax, color='red', alpha=0.5, markersize=0.6, label='Crashes')
sf_speed_camera_locations_meters.plot(ax=ax, color='blue', markersize=30, label='Speed Cameras')
ax.legend()
ax.set_title('Location of Crashes and Speed Cameras')
plt.tight_layout()
plt.savefig('Data/crash_camera_locations.png', dpi=150)
plt.close()

# Count of crashes by distance to the nearest speed camera
crash_by_distance = sf_crashes_dataset_meters[sf_crashes_dataset_meters['distance_to_nearest_camera'] < 3000]
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(crash_by_distance['distance_to_nearest_camera'], bins=20, color='steelblue', edgecolor='white')
ax.set_xlabel('Distance to nearest speed camera (meters)')
ax.set_ylabel('Count of crashes')
ax.set_title('Distribution of Crash Distances to Nearest Camera')
plt.tight_layout()
plt.savefig('Data/crash_distance_distribution.png', dpi=150)
plt.close()

# Count of Speed Camera violations by the year
total_violations = monthly_violations.groupby('year_month').agg(
    sum_violations=('violations_by_month', 'sum')
).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(total_violations['year_month'], total_violations['sum_violations'], color='steelblue')
ax.set_xlabel('Month')
ax.set_ylabel('Violations')
ax.set_title('Speed Camera Violations by Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Data/violations_by_month.png', dpi=150)
plt.close()

# Get crash count by control vs. treatment group
fig, ax = plt.subplots(figsize=(8, 6))
crash_counts = sf_crashes_dataset_meters.groupby('is_within_250_meters').size()
colors = ['gray', 'orange']
ax.bar(crash_counts.index.astype(str), crash_counts.values, color=colors)
ax.set_xlabel('Within 250m (0 = No, 1 = Yes)')
ax.set_ylabel('Crash Count')
ax.set_title('Crashes Within vs Outside 250m of Speed Cameras')
plt.tight_layout()
plt.savefig('Data/crashes_within_outside_250m.png', dpi=150)
plt.close()

################################### Quasi-Experimental Design: Define Control & Testing Groups ###################################

# Control: Crashes outside 250m of a speed camera
control = sf_crashes_dataset_meters.drop(columns='geometry').copy()
control = control[control['outside_250_meters'] == 1]

# Aggregate the Control group
control = control.groupby(['nearest_camera_location_id', 'year_month']).agg(
    total_crash_count=('crash_record_id', 'count'),
    total_injuries=('injuries_total', lambda x: x.fillna(0).sum()),
    total_fatalities=('injuries_fatal', lambda x: x.fillna(0).sum()),
    posted_speed_limit_median=('posted_speed_limit', 'median')
).reset_index()

control = control.rename(columns={'nearest_camera_location_id': 'camera_location_id'})

# Left join to speed camera locations to get go_live_date
camera_dates_control = speed_camera_locations[['location_id', 'go_live_date']].rename(
    columns={'location_id': 'camera_location_id'})
control = control.merge(camera_dates_control, on='camera_location_id', how='left')

# Create is_camera_active and months_since_camera_active columns
control['go_live_date'] = pd.to_datetime(control['go_live_date'])
control['year_month'] = pd.to_datetime(control['year_month'])

control['is_camera_active'] = np.where(
    control['go_live_date'].notna() & (control['year_month'] >= control['go_live_date'].dt.to_period('M').dt.to_timestamp()),
    1, 0)

control['months_since_camera_active'] = np.where(
    control['go_live_date'].notna(),
    ((control['year_month'].dt.year - control['go_live_date'].dt.year) * 12 + 
     (control['year_month'].dt.month - control['go_live_date'].dt.month)),
    np.nan)

# Left join to monthly violations
control = control.merge(monthly_violations, on=['camera_location_id', 'year_month'], how='left')
control['violations_by_month'] = control['violations_by_month'].fillna(0).astype(int)
control['treated'] = 0

# Treatment: Crashes within 250m of a speed camera
treatment = sf_crashes_dataset_meters.drop(columns='geometry').copy()
treatment = treatment[treatment['is_within_250_meters'] == 1]

# Aggregate the Treatment group
treatment = treatment.groupby(['nearest_camera_location_id', 'year_month']).agg(
    total_crash_count=('crash_record_id', 'count'),
    total_injuries=('injuries_total', lambda x: x.fillna(0).sum()),
    total_fatalities=('injuries_fatal', lambda x: x.fillna(0).sum()),
    posted_speed_limit_median=('posted_speed_limit', 'median')
).reset_index()

treatment = treatment.rename(columns={'nearest_camera_location_id': 'camera_location_id'})

# Left join to speed camera locations to get go_live_date
treatment = treatment.merge(camera_dates_control, on='camera_location_id', how='left')

# Create is_camera_active and months_since_camera_active columns
treatment['go_live_date'] = pd.to_datetime(treatment['go_live_date'])
treatment['year_month'] = pd.to_datetime(treatment['year_month'])

treatment['is_camera_active'] = np.where(
    treatment['go_live_date'].notna() & (treatment['year_month'] >= treatment['go_live_date'].dt.to_period('M').dt.to_timestamp()),
    1, 0)

treatment['months_since_camera_active'] = np.where(
    treatment['go_live_date'].notna(),
    ((treatment['year_month'].dt.year - treatment['go_live_date'].dt.year) * 12 + 
     (treatment['year_month'].dt.month - treatment['go_live_date'].dt.month)),
    np.nan)

# Left join to monthly violations
treatment = treatment.merge(monthly_violations, on=['camera_location_id', 'year_month'], how='left')
treatment['violations_by_month'] = treatment['violations_by_month'].fillna(0).astype(int)
treatment['treated'] = 1

# Combine control and treatment groups
panel_all_month = pd.concat([treatment, control], ignore_index=True)
panel_all_month['month'] = panel_all_month['year_month'].dt.month
panel_all_month['year'] = panel_all_month['year_month'].dt.year

# Filter out NAs
panel_all_month = panel_all_month.dropna(subset=['camera_location_id', 'go_live_date', 'months_since_camera_active'])

# Aggregate control and treatment groups by month
panel_summary = panel_all_month.groupby(['treated', 'year_month']).agg(
    total_crash_count=('total_crash_count', 'sum'),
    total_fatalities=('total_fatalities', 'sum'),
    total_injuries=('total_injuries', 'sum'),
    total_violations=('violations_by_month', 'sum'),
    median_speed_limit=('posted_speed_limit_median', 'median')
).reset_index()

panel_summary['group'] = np.where(panel_summary['treated'] == 1, 'Treated (<=250m)', 'Control (>500m)')

# Create pre_post_panel_summary
pre_post_panel_summary = panel_all_month[panel_all_month['go_live_date'].notna()].copy()

pre_post_panel_summary['period'] = np.where(
    pre_post_panel_summary['year_month'] < pre_post_panel_summary['go_live_date'].dt.to_period('M').dt.to_timestamp(),
    'Pre go-live date', 'Post go-live date')

pre_post_panel_summary['group'] = np.where(
    pre_post_panel_summary['treated'] == 1, 'Treated (<=250m)', 'Control (>250m)')

# Aggregate pre/post panel summary
pre_post_agg = pre_post_panel_summary.groupby(['group', 'period']).agg(
    avg_crash_count=('total_crash_count', 'mean'),
    avg_monthly_fatalities=('total_fatalities', 'mean'),
    avg_monthly_injuries=('total_injuries', 'mean'),
    total_monthly_violations=('violations_by_month', 'sum'),
    median_speed_limit=('posted_speed_limit_median', 'median')
).reset_index()

# Get count of average monthly crashes per camera
avg_crashes_per_camera = panel_all_month.groupby(['camera_location_id', 'treated']).agg(
    avg_monthly_crashes=('total_crash_count', 'mean'),
    avg_monthly_fatalities=('total_fatalities', 'mean'),
    avg_monthly_injuries=('total_injuries', 'mean'),
    total_monthly_violations=('violations_by_month', 'sum'),
    median_speed_limit=('posted_speed_limit_median', 'median'),
    n_months=('total_crash_count', 'count')
).reset_index()

################################### Quasi-Experimental Design Graphs ###################################

# Monthly crash counts for the treatment and control groups
fig, ax = plt.subplots(figsize=(14, 8))

for group in panel_summary['group'].unique():
    group_data = panel_summary[panel_summary['group'] == group]
    linestyle = '-' if 'Treated' in group else '--'
    ax.plot(group_data['year_month'], group_data['total_crash_count'], 
            linestyle=linestyle, linewidth=1, label=group)

# Add trendlines using numpy polyfit
for group in panel_summary['group'].unique():
    group_data = panel_summary[panel_summary['group'] == group].sort_values('year_month')
    x_numeric = np.arange(len(group_data))
    z = np.polyfit(x_numeric, group_data['total_crash_count'], 3)
    p = np.poly1d(z)
    ax.plot(group_data['year_month'], p(x_numeric), 
            linestyle=':', color='blue', linewidth=0.7, alpha=0.7)

# Go-live reference line
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
plt.savefig('Data/monthly_crash_counts.png', dpi=150)
plt.close()

# Monthly fatality counts
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
plt.savefig('Data/monthly_fatality_counts.png', dpi=150)
plt.close()

# Plot average monthly crash count per experimental group (Control only)
pre_post_control = pre_post_agg[pre_post_agg['group'] == 'Control (>250m)']

fig, ax = plt.subplots(figsize=(10, 6))
periods = pre_post_control['period'].tolist()
avg_crashes = pre_post_control['avg_crash_count'].tolist()
ax.bar(periods, avg_crashes, color='steelblue')
ax.set_xlabel('Period')
ax.set_ylabel('Average Crash Count')
ax.set_title('Average Monthly Crash Count: Pre vs Post Camera Go-Live')
plt.tight_layout()
plt.savefig('Data/avg_crash_pre_post.png', dpi=150)
plt.close()

################################### Hypothesis Testing ###################################

# H0: Crash rate between Control & Treatment groups is same.
# HA: Crash rate between Control & Treatment groups are different.

print("\n" + "="*80)
print("HYPOTHESIS TESTING")
print("="*80)

################################### Conditional Probability ###################################

panel_conditional_probability = panel_all_month.copy()
panel_conditional_probability['did_fatalities_occur_this_month'] = panel_conditional_probability['total_fatalities'] > 0
panel_conditional_probability['did_injuries_occur_this_month'] = panel_conditional_probability['total_injuries'] > 0

# Calculate the probability of fatalities occurring given experimental groups
fatality_probability_by_groups = panel_conditional_probability.groupby('treated').agg(
    num_camera_months=('treated', 'count'),
    months_with_fatalities=('did_fatalities_occur_this_month', 'sum')
).reset_index()

fatality_probability_by_groups['probability_fatality_given_treat'] = (
    fatality_probability_by_groups['months_with_fatalities'] / 
    fatality_probability_by_groups['num_camera_months'])

print("\nFatality Probability by Groups:")
print(fatality_probability_by_groups)

# Calculate the probability of injuries occurring given experimental groups
injury_probability_by_groups = panel_conditional_probability.groupby('treated').agg(
    num_camera_months=('treated', 'count'),
    months_with_injuries=('did_injuries_occur_this_month', 'sum')
).reset_index()

injury_probability_by_groups['probability_injuries_given_treat'] = (
    injury_probability_by_groups['months_with_injuries'] / 
    injury_probability_by_groups['num_camera_months'])

print("\nInjury Probability by Groups:")
print(injury_probability_by_groups)

################################### Statistical Analysis ###################################

print("\n" + "="*80)
print("STATISTICAL ANALYSIS - T-TESTS")
print("="*80)

# Statistical tests run on raw data
def perform_ttest(data, column, group_col='treated'):
    """Perform two-sided t-test between groups"""
    group0 = data[data[group_col] == 0][column]
    group1 = data[data[group_col] == 1][column]
    return stats.ttest_ind(group0, group1, equal_var=False)

# T-Test on raw data
print("\n--- T-Tests on Raw Data ---")

tt_raw_crashes = perform_ttest(panel_all_month, 'total_crash_count')
print(f"\nT-Test: Total Crash Count")
print(f"  t-statistic: {tt_raw_crashes.statistic:.4f}")
print(f"  p-value: {tt_raw_crashes.pvalue:.6f}")

tt_raw_fatalities = perform_ttest(panel_all_month, 'total_fatalities')
print(f"\nT-Test: Total Fatalities")
print(f"  t-statistic: {tt_raw_fatalities.statistic:.4f}")
print(f"  p-value: {tt_raw_fatalities.pvalue:.6f}")

tt_raw_injuries = perform_ttest(panel_all_month, 'total_injuries')
print(f"\nT-Test: Total Injuries")
print(f"  t-statistic: {tt_raw_injuries.statistic:.4f}")
print(f"  p-value: {tt_raw_injuries.pvalue:.6f}")

tt_raw_violations = perform_ttest(panel_all_month, 'violations_by_month')
print(f"\nT-Test: Violations by Month")
print(f"  t-statistic: {tt_raw_violations.statistic:.4f}")
print(f"  p-value: {tt_raw_violations.pvalue:.6f}")

# Statistical tests run on aggregate data
print("\n--- T-Tests on Aggregate Data ---")

tt_agg_crashes = perform_ttest(avg_crashes_per_camera, 'avg_monthly_crashes')
print(f"\nT-Test: Average Monthly Crashes")
print(f"  t-statistic: {tt_agg_crashes.statistic:.4f}")
print(f"  p-value: {tt_agg_crashes.pvalue:.6f}")

tt_agg_fatalities = perform_ttest(avg_crashes_per_camera, 'avg_monthly_fatalities')
print(f"\nT-Test: Average Monthly Fatalities")
print(f"  t-statistic: {tt_agg_fatalities.statistic:.4f}")
print(f"  p-value: {tt_agg_fatalities.pvalue:.6f}")

tt_agg_injuries = perform_ttest(avg_crashes_per_camera, 'avg_monthly_injuries')
print(f"\nT-Test: Average Monthly Injuries")
print(f"  t-statistic: {tt_agg_injuries.statistic:.4f}")
print(f"  p-value: {tt_agg_injuries.pvalue:.6f}")

tt_agg_violations = perform_ttest(avg_crashes_per_camera, 'total_monthly_violations')
print(f"\nT-Test: Total Monthly Violations")
print(f"  t-statistic: {tt_agg_violations.statistic:.4f}")
print(f"  p-value: {tt_agg_violations.pvalue:.6f}")

################################### Linear Regression ###################################

print("\n" + "="*80)
print("LINEAR REGRESSION ANALYSIS")
print("="*80)

def run_ols_regression(X, y, title):
    """Run OLS regression and print summary"""
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    print(f"\n{title}")
    print("-" * len(title))
    print(model.summary())
    return model

# Only Treated variable included
print("\n--- Model 1: Only Treated Variable ---")
X1 = panel_all_month[['treated']]
y = panel_all_month['total_crash_count']
model1 = run_ols_regression(X1, y, "Model 1: total_crash_count ~ treated")

# treated + fatality rate
print("\n--- Model 2: Treated + Fatalities ---")
X2 = panel_all_month[['treated', 'total_fatalities']]
model2 = run_ols_regression(X2, y, "Model 2: total_crash_count ~ treated + total_fatalities")

# treated + fatality rate + injuries
print("\n--- Model 3: Treated + Fatalities + Injuries ---")
X3 = panel_all_month[['treated', 'total_fatalities', 'total_injuries']]
model3 = run_ols_regression(X3, y, "Model 3: total_crash_count ~ treated + total_fatalities + total_injuries")

# treated + fatality rate + injuries + violations
print("\n--- Model 4: Treated + Fatalities + Injuries + Violations ---")
X4 = panel_all_month[['treated', 'total_fatalities', 'total_injuries', 'violations_by_month']]
model4 = run_ols_regression(X4, y, "Model 4: total_crash_count ~ treated + total_fatalities + total_injuries + violations_by_month")

# Full model with all variables
print("\n--- Model 5: Full Model ---")
# Create interaction term
panel_all_month['camera_active_months_interaction'] = (
    panel_all_month['is_camera_active'] * panel_all_month['months_since_camera_active'].fillna(0))

X5 = panel_all_month[['treated', 'total_injuries', 'posted_speed_limit_median', 
                       'is_camera_active', 'camera_active_months_interaction', 
                       'violations_by_month']].fillna(0)
model5 = run_ols_regression(X5, y, "Model 5: Full Regression Model")

# Determine if crash counts and fatalities are correlated
print("\n--- Crash-Fatality Correlation ---")
X_crash_fatal = panel_all_month[['total_fatalities']]
model_crash_fatal = run_ols_regression(X_crash_fatal, y, "Model: total_crash_count ~ total_fatalities")

################################### Train Test Split ###################################

print("\n" + "="*80)
print("TRAIN/TEST SPLIT EVALUATION")
print("="*80)

# Define a seed for result reproducibility
np.random.seed(123)

# Prepare features for train/test split
features = ['treated', 'total_injuries', 'posted_speed_limit_median', 
            'is_camera_active', 'camera_active_months_interaction', 'violations_by_month']

# Clean data for modeling
model_data = panel_all_month.dropna(subset=features + ['total_crash_count']).copy()

X = model_data[features]
y = model_data['total_crash_count']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train a linear regression model
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train_const).fit()
print("\nModel Summary:")
print(model.summary())

# Make predictions
train_predicted = model.predict(X_train_const)
test_predicted = model.predict(X_test_const)

# Calculate RMSE
def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

train_rmse = rmse(y_train, train_predicted)
test_rmse = rmse(y_test, test_predicted)

print(f"\n--- Model Performance ---")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nPlots saved to Data/ directory:")
print("  - crash_camera_locations.png")
print("  - crash_distance_distribution.png")
print("  - violations_by_month.png")
print("  - crashes_within_outside_250m.png")
print("  - monthly_crash_counts.png")
print("  - monthly_fatality_counts.png")
print("  - avg_crash_pre_post.png")
