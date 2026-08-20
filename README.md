# Speed Camera Effectiveness Analysis

A statistical analysis examining whether speed cameras reduce the frequency of car crashes in Chicago, conducted as part of COSC 6510 Data Intelligence at Marquette University.

## Research Question

**Do speed cameras reduce the frequency of car crashes?**

This study uses a quasi-experimental design to compare crash rates within 250 meters of speed cameras (treatment group) versus crashes occurring beyond this radius (control group).

## Methodology

### Approach
- Draw a 250-meter radius around each speed camera location
- Classify crashes as "treatment" (within radius) or "control" (outside radius)
- Compare crash frequencies, injuries, and fatalities between groups
- Analyze pre vs. post camera activation periods

### Statistical Methods
- Conditional probability analysis
- Two-sample t-tests (raw and aggregated data)
- Linear regression modeling
- Train/test split validation (80/20)

## Datasets

The analysis uses publicly available data from the Chicago Open Data Portal:

| Dataset | Description |
|---------|-------------|
| Speed Camera Locations | Geographic coordinates and activation dates of speed cameras |
| Speed Camera Violations | Monthly violation counts per camera |
| Traffic Crashes - Crashes | Crash records with location, date, and severity |
| Traffic Crashes - People | Information about individuals involved in crashes |
| Traffic Crashes - Vehicles | Vehicle information for each crash |

> **Note:** Large CSV files are not included in this repository due to size constraints. Download them from the [Chicago Data Portal](https://data.cityofchicago.org/).

## Requirements

### R Packages
```r
install.packages(c(
  "ggplot2",
  "tidyverse",
  "janitor",
  "lubridate",
  "sf",
  "MASS",
  "pscl",
  "fixest",
  "Metrics"
))
```

## Project Structure

```
Speed-Camera/
├── main.R                      # Main analysis script
├── main.txt                    # Text backup of analysis
├── Speed_Camera_Locations.csv  # Camera location data (included)
├── Datasets/                   # Additional datasets
└── .gitignore
```

## Key Findings

The analysis examines:
- Monthly crash counts for treatment vs control areas
- Fatality and injury rates by group
- Impact of camera activation on crash frequency
- Correlation between violations and crash rates

## Usage

1. Download the required datasets from the Chicago Data Portal
2. Place CSV files in the project root directory
3. Open `main.R` in RStudio
4. Run the script to reproduce the analysis

## Author

**Nikhil Gajghate**  
M.S. in Computing, Marquette University  
nikhil.gajghate@marquette.edu

## Course

COSC 6510 - Data Intelligence  
Fall 2025  
Marquette University

## License

This project is for academic purposes.
