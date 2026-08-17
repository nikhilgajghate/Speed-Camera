# Marquette University
# Nikhil Gajghate, MSCIS
# nikhil.gajghate@marquette.edu
# COSC 6510 Data Intelligence Final Project
# Final Project: Determine whether Speed Cameras reduce the frequency of car crashes.
# For this study, I'm figuratively drawing an imaginary circle with radius 250m 
# around a camera and making a note of accidents that occurs within said 
# circle vs. outside to conduct this analysis.

# clear the working directory
rm(list=ls())

################################### Package Installation ###################################

# install.packages("read_xl")
# install.packages("ggplot2")
# install.packages("tidyverse")
# install.packages("janitor")
# install.packages("lubridate")
# install.packages("sf")
# install.packages("MASS")
# install.packages("pscl")
# install.packages("fixest")
# install.packages("Metrics")

################################### Import libraries ###################################

library(ggplot2)
library(dplyr)
library(tidyverse)
library(janitor)
library(lubridate)
library(sf)
library(MASS)
library(pscl)
library(fixest)
library(Metrics)

################################### Import datasets ###################################

# Read Speed Camera Locations dataset
speed_camera_locations <- read.csv('Speed_Camera_Locations.csv')
dim(speed_camera_locations)
# View(speed_camera_locations[0:100,])

# Read Speed Camera Violations dataset
speed_camera_violations <- read.csv('Speed_Camera_Violations.csv')
dim(speed_camera_violations)
# View(speed_camera_violations[0:100,])

# Read Traffic Crashes - Crashes dataset
crashes_dataset <- read.csv('Traffic_Crashes_-_Crashes.csv')
dim(crashes_dataset)
# View(crashes_dataset[0:100,])

# Read Traffic Crashes - People dataset
people_dataset <- read.csv('Traffic_Crashes_-_People.csv')
dim(people_dataset)
# View(people_dataset[0:100,])

# Read Traffic Crashes - Vehicles dataset
vehicles_dataset <- read.csv('Traffic_Crashes_-_Vehicles.csv')
dim(vehicles_dataset)
# View(vehicles_dataset[0:100,])

################################### Data Wrangling & Feature Engineering ###################################

# Convert all columns from UPPER CASE to snake_case using clean_names()
# Example Usage (Speed Camera Locations Dataset)
# Before: "ID", "LOCATION.ID", "ADDRESS", "FIRST.APPROACH", "SECOND.APPROACH", "GO.LIVE.DATE", "LATITUDE", "LONGITUDE", "LOCATION"
# After: "id", "location_id", "address", "first_approach", "second_approach", "go_live_date", "latitude", "longitude", "location"
speed_camera_locations <- speed_camera_locations |> clean_names()
speed_camera_violations <- speed_camera_violations |> clean_names()
crashes_dataset <- crashes_dataset |> clean_names()
people_dataset <- people_dataset |> clean_names()
vehicles_dataset <- vehicles_dataset |> clean_names()

# Remove redundant columns
# speed_camera_locations: drop column second_approach because it is not being used. 
colSums(is.na(speed_camera_locations))
speed_camera_locations$second_approach <- NULL
colSums(is.na(speed_camera_locations))


# people_dataset: drop columns seat_no, drivers_license_class, ems_agency, 
# hospital, ems_run_no, bac_result_value because they are not being used. 
colSums(is.na(people_dataset))
people_dataset$seat_no <- NULL
people_dataset$drivers_license_class <- NULL
people_dataset$ems_agency <- NULL
people_dataset$hospital <- NULL
people_dataset$ems_run_no <- NULL
people_dataset$bac_result_value <- NULL
colSums(is.na(people_dataset))

# vehicles_dataset: drop columns trailer1_length, trailer2_length, 
# total_vehicle_length, ccmc_no, ilcc_no, idot_permit_no, towed_by, towed_to, 
# num_passengers because they are not being used. 
colSums(is.na(vehicles_dataset))
vehicles_dataset$trailer1_length <- NULL
vehicles_dataset$trailer2_length <- NULL
vehicles_dataset$total_vehicle_length <- NULL
vehicles_dataset$ccmc_no <- NULL
vehicles_dataset$ilcc_no <- NULL
vehicles_dataset$idot_permit_no <- NULL
vehicles_dataset$towed_by <- NULL
vehicles_dataset$towed_to <- NULL
vehicles_dataset$num_passengers <- NULL
vehicles_dataset$vehicle_year <- NULL
colSums(is.na(vehicles_dataset))


# Drop NAs from speed_camera_violations
# speed_camera_violations: drop NAs from x_coordinate, y_coordinate, latitude, longitude
dim(speed_camera_violations)
speed_camera_violations <- speed_camera_violations |> 
  filter(!is.na(x_coordinate) & !is.na(y_coordinate) & !is.na(latitude) & !is.na(longitude))
dim(speed_camera_violations)


# Drop NAs from crashes_dataset
colSums(is.na(crashes_dataset)) # drop NAs from injuries_total, injuries_fatal, 
# injuries_incapacitating, injuries_non_incapcitating, injuries_reported_not_evident, 
# injuries_no_indication, injuries_unknown, num_people
dim(crashes_dataset)
crashes_dataset <- crashes_dataset |> 
  filter(!is.na(injuries_total) & !is.na(injuries_fatal) & !is.na(injuries_incapacitating)
         & !is.na(injuries_non_incapacitating) & !is.na(injuries_reported_not_evident) 
         & !is.na(injuries_no_indication) & !is.na(injuries_unknown) 
         & !is.na(latitude) & !is.na(longitude))
dim(crashes_dataset)
colSums(is.na(crashes_dataset))


# Drop NAs from vehicles_dataset
colSums(is.na(vehicles_dataset))
dim(vehicles_dataset)
vehicles_dataset <- vehicles_dataset |> 
  filter(!is.na(vehicle_id))
dim(vehicles_dataset)
colSums(is.na(vehicles_dataset))


# Convert date format to preferred type. See inline comments for more info.
crashes_dataset <- crashes_dataset |> 
  mutate(crash_datetime = mdy_hms(crash_date, tz = "America/Chicago"), # create a new variable called crash_datetime to store the contents of crash_date 
         crash_date = as_date(crash_datetime), # Convert crash_date to only contain yyyy-mm-dd. "09/11/2025 12:18:00 AM" -> "2025-09-11"
         year_month = floor_date(crash_datetime, "month")) # create a new variable called year_month that stores the year and the month this crash took place and disregards the date

# Convert date format from MM/DD/YYYY to YYYY-MM-DD
speed_camera_locations <- speed_camera_locations |> 
  mutate(go_live_date = mdy(go_live_date))

# Convert date format from MM/DD/YYYY to YYYY-MM-DD. See inline comments for 
# more info. 
speed_camera_violations <- speed_camera_violations |> 
  rename(camera_location_id = camera_id) |> # Rename camera_id to camera_location_id because it matches location_id in the speed_camera_locations dataset
  mutate(violation_date = mdy(violation_date), # Convert date format from MM/DD/YYYY to YYYY-MM-DD
         year_month = floor_date(violation_date, "month")) # create a new variable called year_month that stores the year and the month this crash took place and disregards the date 

# Convert the violations column in speed_camera_violations from character to numeric type
class(speed_camera_violations$violations)
speed_camera_violations$violations <- as.numeric(speed_camera_violations$violations)
class(speed_camera_violations$violations)

# The above conversion results in some NAs
sum(is.na(speed_camera_violations$violations)) # 38 NAs

# Filters violations where the count of violations is NA
# Before
dim(speed_camera_violations)
speed_camera_violations <- speed_camera_violations |> 
  filter(!is.na(violations))
dim(speed_camera_violations)

# Left join crashes, vehicles and people dataset on crash_record_id column 
# for feature engineering
crashes_dataset <- crashes_dataset |> 
  left_join(people_dataset |> 
              count(crash_record_id, name='num_people'), # Create a new variable called num_people that stores the number of people involved in a crash
            by="crash_record_id") |>
  left_join(vehicles_dataset |> 
              count(crash_record_id, name='num_vehicles'), # Create a new variable called num_vehicles that stores the number of vehicles involved in a crash
            by="crash_record_id")

# Convert coordinates columns into sf objects
# https://stackoverflow.com/questions/59183384/how-to-convert-x-and-y-coordinates-into-latitude-and-longitude
sf_crashes_dataset <- crashes_dataset |> 
  st_as_sf(coords=c("longitude", "latitude"), # specify longitude and latitude
           crs=4326, # WGS84 Coordinate Reference System
           remove=FALSE)

# Note: Speed camera locations latitudes and longitudes are not null.
sf_speed_camera_locations_dataset <- speed_camera_locations |>
  st_as_sf(coords=c("longitude", "latitude"), # specify longitude and latitude
           crs=4326, # WGS84 Coordinate Reference System
           remove=FALSE)

# Convert the coordinates to the units used in IL so that distance now is in Meters and not degrees
sf_crashes_dataset_meters <- st_transform(sf_crashes_dataset, 3435)
sf_speed_camera_locations_meters <- st_transform(sf_speed_camera_locations_dataset, 3435)

# Find the index of the nearest camera for every crash
nearest_index <- st_nearest_feature(sf_crashes_dataset_meters, sf_speed_camera_locations_meters)

# Get the camera id for the nearest camera and find out how far away it is in meters
sf_crashes_dataset_meters <- sf_crashes_dataset_meters |> 
  mutate(nearest_camera_location_id=sf_speed_camera_locations_meters$location_id[nearest_index],
         distance_to_nearest_camera=as.numeric(st_distance(geometry, 
                                                           sf_speed_camera_locations_meters$geometry[nearest_index],
                                                           by_element=TRUE)))

# Join crashes to the speed camera locations dataset to get the 
# date the nearest camera went live 
sf_crashes_dataset_meters <- sf_crashes_dataset_meters |>
  left_join(speed_camera_locations |> dplyr::select(location_id, go_live_date),
            by=c("nearest_camera_location_id"="location_id"))

# Create a new column called is_crash_camera_active that is:
# 1 if the crash_date is after the live date for the camera else 0
sf_crashes_dataset_meters <- sf_crashes_dataset_meters |>
  mutate(is_crash_camera_active=if_else(!is.na(go_live_date) & crash_date >= go_live_date, 1L, 0L))


# Define thresholds
threshold_250 <- st_buffer(sf_speed_camera_locations_meters, dist=250)

# Based on thresholds, create two new columns named is_within_250_meters vs.
# outside_250_meters based on the following conditions:
# if the accident took place within the threshold vs. outside
sf_crashes_dataset_meters <- sf_crashes_dataset_meters |>
  mutate(is_within_250_meters=as.integer(lengths(st_within(sf_crashes_dataset_meters, threshold_250)) > 0),
         outside_250_meters=as.integer(lengths(st_within(sf_crashes_dataset_meters, threshold_250)) == 0))

# Create a monthly violations variable to get the count of violations by month.
monthly_violations <- speed_camera_violations |>
  group_by(camera_location_id, year_month) |>
  summarise(violations_by_month=sum(violations, na.rm=TRUE), .groups="drop")

################################### Data Wrangling Graphs ###################################

# Plot the location of the crash and the cameras
ggplot() +
  geom_sf(data = sf_crashes_dataset_meters[0:10000,], aes(color="Crashes"), alpha=0.5, size=0.6) +
  geom_sf(data = sf_speed_camera_locations_dataset, aes(color="Speed Cameras"), size=3) +
  scale_color_manual(values=c("Crashes"="red", "Speed Cameras"="blue")) +
  theme_minimal() +
  labs(title="Location of Crashes and Speed Cameras",
       color="Legend")

# Count of crashes by distance to the nearest speed camera. Cut it off at 3000m
# to help narrate the problem statement.
crash_by_distance <- sf_crashes_dataset_meters[sf_crashes_dataset_meters$distance_to_nearest_camera < 3000, ]
ggplot(crash_by_distance, aes(x = distance_to_nearest_camera)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "white") +
  theme_minimal() +
  labs(title="Distribution of Crash Distances to Nearest Camera",
       x="Distance to nearest speed camera (meters)",
       y="Count of crashes")


# Count of Speed Camera violations by the year. Interesting graph because
# spike after Covid
total_violations <- monthly_violations %>%
  group_by(year_month) %>%
  summarise(sum_violations = sum(violations_by_month))
ggplot(total_violations, aes(x=year_month, y=sum_violations)) +
  geom_bar(stat='identity', fill='steelblue') +
  geom_smooth(method='loess', size=1.2, color="darkred")
  theme_minimal() +
  labs(title="Speed Camera Violations by Month",
       x="Month",
       y="Violations")
 
 
# Get crash count by control vs. treatment group
ggplot(sf_crashes_dataset_meters,
       aes(x = factor(is_within_250_meters), fill = factor(is_within_250_meters))) +
  geom_bar() +
  scale_fill_manual(values=c("0"="gray", "1"="orange")) +
  theme_minimal() +
  labs(title="Crashes Within vs Outside 250m of Speed Cameras",
       x="Within 250m (0 = No, 1 = Yes)",
       y="Crash Count",
       fill="Within 250m Indicator")


################################### Quasi-Experimental Design: Define Control & Testing Groups ###################################
# Control: Crashes outside 250m of a speed camera
control <- sf_crashes_dataset_meters |>
  st_drop_geometry() # Drop the geometry column that was created as part of st_as_sf()


# Filter the Control group to only include data on crashes outsides 250 meters
control <- control |>
  filter(outside_250_meters == 1) 

# Aggregate the Control group to get: 
# total number of crash counts, total number of injuries, total number of fatalities
# by months of the year
control <- control |>
  group_by(nearest_camera_location_id, year_month) |>
  summarise(total_crash_count = n(),
            total_injuries = sum(injuries_total %||% 0, na.rm = TRUE),
            total_fatalities = sum(injuries_fatal %||% 0, na.rm = TRUE),
            posted_speed_limit_median = median(posted_speed_limit, na.rm = TRUE),
            .groups = "drop")

# Rename the nearest_camera_location_id column to camera_location_id
control <- control |>
  rename(camera_location_id = nearest_camera_location_id)

# Left join to speed camera locations to get access to go_live_date
control <- control |>
  left_join(speed_camera_locations |> dplyr::select(location_id, go_live_date) |>
              rename(camera_location_id = location_id),
            by = "camera_location_id")

# Create 2 new columns called:
# is_camera_active: to represent whether the camera was active or not
# months_since_camera_active: to represent the number of months since the camera has been active
control <- control |>
  mutate(is_camera_active = if_else(!is.na(go_live_date) & year_month >= floor_date(go_live_date, "month"),
                                    1L, 0L),
         months_since_camera_active = if_else(!is.na(go_live_date),
                                              interval(go_live_date, year_month) %/% months(1),
                                              NA_integer_))

# left join to monthly violations to get count of violations
control <- control |>
  left_join(monthly_violations,
            by = c("camera_location_id", "year_month"))

# Replace NAs with 0
control <- control |>
  mutate(violations_by_month = replace_na(violations_by_month, 0L),
         treated = 0L)

# Treatment: Crashes within 250m of a speed camera
treatment <- sf_crashes_dataset_meters |>
  st_drop_geometry() # Drop the geometry column that was created as part of st_as_sf()
  
# Filter the Treatment group to only include data on crashes within 250 meters  
treatment <- treatment |>  
  filter(is_within_250_meters == 1)

# Aggregate the Control group to get: 
# total number of crash counts, total number of injuries, total number of fatalities, median posted speed limit
# by months of the year
treatment <- treatment |>  
  group_by(nearest_camera_location_id, year_month) |>
  summarise(total_crash_count = n(),
            total_injuries = sum(injuries_total %||% 0, na.rm = TRUE),
            total_fatalities = sum(injuries_fatal %||% 0, na.rm = TRUE),
            posted_speed_limit_median = median(posted_speed_limit, na.rm = TRUE),
            .groups = "drop")

# Rename the nearest_camera_location_id column to camera_location_id
treatment <- treatment |>
  rename(camera_location_id = nearest_camera_location_id)
  
# Left join to speed camera locations to get access to go_live_date
treatment <- treatment |>  
  left_join(speed_camera_locations |> dplyr::select(location_id, go_live_date) |> 
              rename(camera_location_id = location_id),
            by = "camera_location_id") 

# Create 2 new columns called:
# is_camera_active: to represent whether the camera was active or not
# months_since_camera_active: to represent the number of months since the camera has been active
treatment <- treatment |>  
  mutate(is_camera_active = if_else(!is.na(go_live_date) & year_month >= floor_date(go_live_date, "month"),
                                 1L, 0L),
         months_since_camera_active = if_else(!is.na(go_live_date),
                                       interval(go_live_date, year_month) %/% months(1),
                                       NA_integer_)) 

# left join to monthly violations to get count of violations 
treatment <- treatment |>
  left_join(monthly_violations, 
            by = c("camera_location_id", "year_month")) 

# Replace NAs with 0
treatment <- treatment |>
  mutate(violations_by_month = replace_na(violations_by_month, 0L),
         treated = 1L)

# Combine control and treatment groups into one dataframe
panel_all_month <- bind_rows(treatment, control) |>
  mutate(month = month(year_month), year = year(year_month))

# Convert year_month and go_live_date into Date type
panel_all_month <- panel_all_month %>%
  mutate(
    year_month = as.Date(year_month), # parse YYYY-MM-DD (month start)
    go_live_date = as.Date(go_live_date) # parse camera go-live if present
  ) %>%
  filter(!is.na(camera_location_id) & !is.na(go_live_date) & !is.na(months_since_camera_active))

# Aggregate control and treatment groups by month to get the total count of 
# fatalities and crashes
panel_summary <- panel_all_month %>%
  group_by(treated, year_month) %>%
  summarise(
    total_crash_count = sum(total_crash_count, na.rm = TRUE),
    total_fatalities = sum(total_fatalities, na.rm = TRUE),
    total_injuries = sum(total_injuries, na.rm = TRUE),
    total_violations = sum(violations_by_month, na.rm = TRUE),
    median_speed_limit = median(posted_speed_limit_median, na.rm = TRUE), 
    .groups = "drop"
    ) %>%
  mutate(group = if_else(treated == 1, "Treated (<=250m)", "Control (>500m)"))

# Create a new variable called pre_post_panel_summary to include
# crash, fatality, speed details for both the treatment groups before vs. after
# camera activation and filter out any NAs in the go_live_date
pre_post_panel_summary <- panel_all_month %>%
  filter(!is.na(go_live_date))

# Add 2 new columns to the dataframe to capture pre-camera vs. post-camera data
pre_post_panel_summary <- pre_post_panel_summary %>%
  mutate(
    period = if_else(year_month < floor_date(go_live_date, "month"),
                     "Pre go-live date", "Post go-live date"),
    group = if_else(treated == 1, "Treated (<=250m)", "Control (>250m)")
  ) 


# Rearrange the order of the columns
pre_post_panel_summary$period <- factor(
  pre_post_panel_summary$period,
  levels = c("Pre go-live date", "Post go-live date")
)

# Aggregate the control vs. treatments groups by the newly created
# pre vs. post camera columns to get average crash counts for the two groups 
# before & after the camera went live. 
pre_post_panel_summary <- pre_post_panel_summary %>%
  group_by(group, period) %>%
  summarise(
    avg_crash_count = mean(total_crash_count, na.rm = TRUE),
    avg_monthly_fatalities = mean(total_fatalities, na.rm = TRUE),
    avg_monthly_injuries = mean(total_injuries, na.rm = TRUE),
    total_monthly_violations = sum(violations_by_month, na.rm = TRUE),
    median_speed_limit = median(posted_speed_limit_median, na.rm = TRUE),
    .groups = "drop"
  )

# Filter pre-post panel summary to only include Control data
pre_post_panel_summary <- pre_post_panel_summary %>%
  filter(group == 'Control (>250m)')

# Get count of average monthly crashes per camera for both control & treatment
avg_crashes_per_camera <- panel_all_month %>%
  group_by(camera_location_id, treated) %>%
  summarise(
    avg_monthly_crashes = mean(total_crash_count, na.rm = TRUE),
    avg_monthly_fatalities = mean(total_fatalities, na.rm = TRUE),
    avg_monthly_injuries = mean(total_injuries, na.rm = TRUE),
    total_monthly_violations = sum(violations_by_month, na.rm = TRUE),
    median_speed_limit = median(posted_speed_limit_median, na.rm = TRUE),
    n_months = n(),
    .groups = "drop"
  )

################################### Quasi-Experimental Design Graphs ###################################

# Monthly crash counts for the treatment and control groups
ggplot(panel_summary, aes(x = year_month, y = total_crash_count, linetype = group)) +
  geom_line(linewidth = 1) +
  # Trendlines (one per group)
  geom_smooth(
    aes(group = group),
    method = "gam", # Linear trendline
    se = FALSE, # Hide confidence band
    color = "blue", # Neutral color for clarity
    linetype = "dashed", # different from main line
    linewidth = 0.7
  ) +
  # Go-live reference line
  geom_vline(
    xintercept = as.Date('2015-07-01'),
    color = "red",
    linetype = "dashed",
    linewidth = 0.8) +
  # Optional label for the spike line
  annotate(
    "text",
    x = as.Date('2015-07-01'),
    y = max(panel_summary$total_crash_count, na.rm = TRUE) * 0.5,
    label = 'More consistent data post 07-01-2015',
    color = "red",
    angle = 90,
    vjust = -0.5,
    hjust = 0,
    size = 3.5
  ) +
  # Add title, subtitle, axis labels, and legend
  labs(
    title = "Monthly Crash Counts: Treated vs Control Areas",
    subtitle = "\nControl: Crashes reported >500m of speed cameras\n\nTreatment: Crashes reported <=250m of speed cameras",
    x = "Year",
    y = "Crash Count",
    linetype = "Group",
    caption = "Data Source: City Open Data Portal | Analysis by Nikhil Gajghate"
  ) +
  # A clean, professional theme
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "gray30"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )


# Monthly fatality counts for the treatment and control groups
ggplot(panel_summary,aes(x = year_month, y = total_fatalities, linetype = group)) +
  geom_line(linewidth = 1) +
  # Trendlines (one per group)
  geom_smooth(
    aes(group = group),
    method = "gam", # Linear trendline
    se = FALSE, # Hide confidence band
    color = "blue", # Neutral color for clarity
    linetype = "dashed", # different from main line
    linewidth = 0.7
  ) +
  # Go-live reference line
  geom_vline(
    xintercept = as.Date('2015-07-01'),
    color = "red",
    linetype = "dashed",
    linewidth = 0.8
  ) +
  # Optional label for the spike line
  annotate(
    "text",
    x = as.Date('2015-07-01'),
    y = max(panel_summary$total_fatalities, na.rm = TRUE) * 0.5,
    label = 'More consistent data post 07-01-2015',
    color = "red",
    angle = 90,
    vjust = -0.5,
    hjust = 0,
    size = 3.5
  ) +
  # Add title, subtitle, axis labels, and legend
  labs(
    title = "Monthly Fatality counts: Treated vs Control Areas",
    subtitle = "\nControl: Crashes reported >250m of speed cameras\n\nTreatment: Crashes reported <=250m of speed cameras",
    x = "Year",
    y = "Fatalities Count",
    linetype = "Group",
    caption = "Data Source: City Open Data Portal | Analysis by Nikhil Gajghate"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )
  
# Plot Monthly crash count for the Control group pre vs. post-deployment
ggplot(pre_post_panel_summary, aes(x = period, y = avg_crash_count, fill = group)) +
  geom_col(position = "dodge") +
  theme_minimal() +
  labs(
    title = "Average Monthly Crash Count: Pre vs Post Camera Go-Live",
    x = "Period",
    y = "Average Crash Count",
    fill = "Group"
  )

################################### Hypothesis Testing ###################################

# H0: Crash rate between Control & Treatment groups is same. 
# HA: Crash rate between Control & Treatment groups are different.

################################### Conditional Probability ###################################

panel_conditional_probability <- panel_all_month |>
  mutate(
    did_fatalities_occur_this_month = total_fatalities > 0,
    did_injuries_occur_this_month = total_injuries > 0,
    did_crashes_occur_this_month = total_crash_count > 0,
  )

# Calculate the probability of fatalities occurring given experimental groups
fatality_probability_by_groups <- panel_conditional_probability |> 
  group_by(treated) |>
  summarise(
    num_camera_months = n(),
    months_with_fatalities = sum(did_fatalities_occur_this_month, na.rm = TRUE),
    probability_fatality_given_treat = months_with_fatalities/num_camera_months,
    .groups = "drop"
  )

fatality_probability_by_groups

# Calculate the probability of injuries occurring given experimental groups
injury_probability_by_groups <- panel_conditional_probability |> 
  group_by(treated) |>
  summarise(
    num_camera_months = n(),
    months_with_injuries = sum(did_injuries_occur_this_month, na.rm = TRUE),
    probability_injuries_given_treat = months_with_injuries/num_camera_months,
    .groups = "drop"
  )

injury_probability_by_groups


################################### Statistical Analysis ###################################

# Statistical tests run on raw data
tt_raw_crashes <- t.test(
  total_crash_count ~ treated,
  data = panel_all_month,
  alternative = "two.sided"
)
tt_raw_crashes

tt_raw_fatalities <- t.test(
  total_fatalities ~ treated,
  data = panel_all_month,
  alternative = "two.sided"
)
tt_raw_fatalities

tt_raw_injuries <- t.test(
  total_injuries ~ treated,
  data = panel_all_month,
  alternative = "two.sided"
)
tt_raw_injuries

tt_raw_violations <- t.test(
  violations_by_month ~ treated,
  data = panel_all_month,
  alternative = "two.sided"
)
tt_raw_violations

# Statistical tests run on aggregate data
# T-Test: Average Monthly crashes between the two experimental groups
tt_agg_crashes <- t.test(
  avg_monthly_crashes ~ treated,
  data = avg_crashes_per_camera,
  alternative = "two.sided"
)
tt_agg_crashes

# T-Test: Average Monthly fatalities between the two experimental groups
tt_agg_fatalities <- t.test(
  avg_monthly_fatalities ~ treated,
  data = avg_crashes_per_camera,
  alternative = "two.sided"
)
tt_agg_fatalities

# T-Test: Average Monthly injuries between the two experimental groups
tt_agg_injuries <- t.test(
  avg_monthly_injuries ~ treated,
  data = avg_crashes_per_camera,
  alternative = "two.sided"
)
tt_agg_injuries

# T-Test: Average Monthly violations between the two experimental groups
tt_agg_violations <- t.test(
  total_monthly_violations ~ treated,
  data = avg_crashes_per_camera,
  alternative = "two.sided"
)
tt_agg_violations

################################### Linear Regression ###################################

# Only Treated variable included
only_treated <- lm(
  total_crash_count ~ treated,
  data = panel_all_month,
)
summary(only_treated)

# treated + fatality rate
treated_fatal <- lm(
  total_crash_count ~ treated + total_fatalities,
  data = panel_all_month,
)
summary(treated_fatal)

# treated + fatality rate + injuries
treated_fatal_injuries <- lm(
  total_crash_count ~ treated + total_fatalities + total_injuries,
  data = panel_all_month,
)
summary(treated_fatal_injuries)

# treated + fatality rate + injuries + violations
treated_fatal_injuries_violations <- lm(
  total_crash_count ~ treated + total_fatalities + total_injuries + violations_by_month,
  data = panel_all_month,
)
summary(treated_fatal_injuries_violations)

# Linear Regression on all variables
regression <- lm(
  total_crash_count ~ treated +
                     total_injuries +
                     posted_speed_limit_median +
                     is_camera_active +
                     (is_camera_active * months_since_camera_active) +
                     violations_by_month,
  data = panel_all_month
)
summary(regression)

# Determine if crash counts and fatalities are correlated
crash_fatality <- lm(
  total_crash_count ~ total_fatalities,
  data = panel_all_month,
)
summary(crash_fatality)

################################### Optional: Train Test Split ###################################
# https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function

# Define a seed for result reproducibility
set.seed(123)

# Define training size
train_size <- floor(0.8 * nrow(panel_all_month))
train_idx <- sample(seq_len(nrow(panel_all_month)), size=train_size)

# Slice the data based on the training size to get both train and test sets
train <- panel_all_month[train_idx, ]
test <- panel_all_month[-train_idx, ]

# Train a linear regression model
model <- lm(
  total_crash_count ~ treated +
    total_injuries +
    posted_speed_limit_median +
    is_camera_active +
    (is_camera_active * months_since_camera_active) +
    violations_by_month,
  data = train
)

summary(model)

# Make predictions on train data so that we can get training metrics
train$predicted <- predict(model, newdata = train)

# Make predictions on test data
test$predicted <- predict(model, newdata = test)


# Define Root Mean Squared Error as the evaluation function
# https://www.r-bloggers.com/2021/07/how-to-calculate-root-mean-square-error-rmse-in-r/
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

# Compute Training RMSE
train_rmse <- rmse(train$total_crash_count, train$predicted)
print(paste('Training Accuracy:', round(train_rmse, 2)))

# Compute Testing RMSE
test_rmse <- rmse(test$total_crash_count, test$predicted)
print(paste('Testing Accuracy:', round(test_rmse, 2)))












