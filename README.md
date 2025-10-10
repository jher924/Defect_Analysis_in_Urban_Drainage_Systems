# Defect_Analysis_in_Urban_Drainage_Systems

This repository contains the code developed for the **Distribution and Properties of Defects in Urban Drainage Systems: An analysis of Auckland's Sewer Network** paper. Juana Herrán, María A. González, Jakobus E. van Zyl, and Theunis F. P. Henning. 

**Email corresponding author:** jher924@aucklanduni.ac.nz

The code allows users to:

- Describe a sewer network and compare it with the subset of pipes inspected via CCTV, assessing representativeness.
- Analyze and visualize properties of defects observed during inspections, including:
  - Average number of defects per km and per pipe
  - Defect type
  - Defect size
  - Longitudinal distance
  - Extent
  - Circumferential position
---

## 1) Overview

To run the code with your own data, prepare an input file containing three sheets:

1. **PIPES** – Description of the pipes in the network.
2. **CCTV** – Information about the pipes that have been inspected.
3. **DEFECTS** – Details of observed defects.

Each sheet must follow the required column structure (see details below).


It is recommended to check the example workbook **`PIPE_GITHUB.xlsx`**, This file does not contain real data, nor is it the data used in the paper, as it cannot be shared due to confidentiality. However, the file serves as a reference for the format and the columns required to run the code.

<table style="text-align:center;">
  <tr>
    <th>Sheet</th>
    <th>Required Column</th>
    <th>Description of the Column</th>
  </tr>

  <!-- PIPES -->
  <tr>
    <td rowspan="13" style="vertical-align:middle;">PIPES</td>
    <td>PIPE_ID</td>
    <td>Unique identifier for each pipe in the network.</td>
  </tr>
  <tr>
    <td>MATERIAL</td>
    <td>Pipe material (e.g., PVC, PE, AC, CONC, VC).</td>
  </tr>
  <tr>
    <td>DIAMETER</td>
    <td>Internal pipe diameter (mm). Must be numeric, integer.</td>
  </tr>
  <tr>
    <td>LENGTH</td>
    <td>Pipe length (m). Must be numeric and non-negative.</td>
  </tr>
  <tr>
    <td>SLOPE</td>
    <td>Pipe slope, usually expressed as rise/run or %.</td>
  </tr>
  <tr>
    <td>AVG_DEPTH</td>
    <td>Average depth of the pipe below the surface (m).</td>
  </tr>
  <tr>
    <td>INSTALL_YEAR</td>
    <td>Year the pipe was installed. Must be a 4-digit integer.</td>
  </tr>
  <tr>
    <td>FLOW_DRY_MAX</td>
    <td>Maximum dry-weather flow capacity (L/s).</td>
  </tr>
  <tr>
    <td>PIPE_CAPACITY</td>
    <td>Nominal design capacity of the pipe (L/s).</td>
  </tr>
  <tr>
    <td>FLOW_WET_MAX</td>
    <td>Maximum wet-weather flow (L/s).</td>
  </tr>
  <tr>
    <td>SEWER_CATEGORY</td>
    <td>Classification of the pipe within the sewer network (e.g., local, transmission).</td>
  </tr>
  <tr>
    <td>SEWAGE_CATEGORY</td>
    <td>Type of sewage conveyed (e.g., wastewater, combined, stormwater).</td>
  </tr>
  <tr>
    <td>LINING</td>
    <td>Type of internal lining applied to the pipe (if any).</td>
  </tr>

  <!-- Separator -->
  <tr><td colspan="3"><hr></td></tr>

  <!-- CCTV -->
  <tr>
    <td rowspan="4" style="vertical-align:middle;">CCTV</td>
    <td>PIPE_ID</td>
    <td>Unique identifier linking each CCTV inspection to the corresponding pipe in `df_pipes`.</td>
  </tr>
  <tr>
    <td>INSP_DIRECTION</td>
    <td>Inspection direction, usually indicating whether the survey was carried out upstream or downstream.</td>
  </tr>
  <tr>
    <td>INSP_DATE</td>
    <td>Date when the CCTV inspection was performed (recommended format: YYYY-MM-DD).</td>
  </tr>
  <tr>
    <td>SURVEY_LENGTH</td>
    <td>Length of the pipe surveyed during the CCTV inspection (m).</td>
  </tr>

  <!-- Separator -->
  <tr><td colspan="3"><hr></td></tr>

  <!-- DEFECTS -->
  <tr>
    <td rowspan="9" style="vertical-align:middle;">DEFECTS</td>
    <td>DEFECT_ID</td>
    <td>Unique identifier for each defect.</td>
  </tr>
  <tr>
    <td>PIPE_ID</td>
    <td>Unique identifier for each pipe in the network.</td>
  </tr>
  <tr>
    <td>DEFECT_TYPE</td>
    <td>Type of defect.</td>
  </tr>
  <tr>
    <td>DEFECT_SIZE</td>
    <td>Size of the defect. Must be classified as S, M, or L. If the size is in a different format, it is recommended to adjust it to these categories in order to run the size plot.</td>
  </tr>
  <tr>
    <td>CLOCK_REFERENCE_START</td>
    <td>Circumferential position where the defect begins. This is given by a clock reference (integer 1–12). If 0, the defect has no circumferential extent.</td>
  </tr>
  <tr>
    <td>CLOCK_REFERENCE_END</td>
    <td>Circumferential position where the defect ends. This is given by a clock reference (integer 1–12). If 0, the defect has no circumferential extent.</td>
  </tr>
  <tr>
    <td>LONGITUDINAL_DISTANCE_NORMALIZED</td>
    <td>Normalized longitudinal distance of the defect (defect position along the pipe divided by the pipe length). It represents the starting point on extend defects</td>
  </tr>
  <tr>
    <td>DEFECT_LENGTH</td>
    <td>Normalized defect length (distance from the start of the defect to its end, relative to pipe length).</td>
  </tr>
<table>


## 2) Code Structure

The code is organized into five main sections, described below:

#### Section 1: Installation and Setup
This section installs all the necessary packages required to run the code.
#### Section 2: Data Input and Validation Workflow
In this section, the user provides the input data.
The code validates the data and generates an Excel report highlighting any errors or warnings in the input.
#### Section 3: Data Preparation
This section prepares the data for analysis.
It performs necessary merges. The user selects the materials and variables of interest for further processing.
#### Section 4: Dataset Description
The code provides a detailed description of the dataset, summarizing key properties and characteristics of the sewer network and inspections. It also analyzes the representativeness of the pipes inspected by CCTV.
#### Section 5: Defect Properties Analysis
This section analyzes the properties of observed defects.



## 3) Installation and Setup
**Required Dependencies**

Before using the code, ensure you have the following packages installed:
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical plotting
- `statsmodels`: Statistical analysis
- `scipy`: Statistical tests
- `tkinter`: File dialogs and simple GUI
- `openpyxl`: Excel file support



## 4) Data Input and Validation Workflow
**Input Data**

The validator is designed to work with a single Excel workbook that must contain **three sheets**: `PIPES`, `CCTV`, and `DEFECTS`

If a CSV file is provided instead, it will be treated as the `PIPES` sheet only; the other two will remain empty.

**Output Report**

The script creates an Excel report with four sheets:
- **SUMMARY**: Number of errors and warnings per input sheet
- **PIPES / CCTV / DEFECTS**: Detailed validation results

The validator checks for:
- Missing or null values
- Non-numeric or negative numbers in numeric fields
- Duplicated identifiers
- Out-of-range installation years or diameters



## 5) Data Preparation

This section merges and prepares the data for analysis.

- **Select Material(s) to Analyze _(User can edit)_**: Choose one or more materials to include (e.g., AC, CONC, VC, PVC, PE).

- **Select the factor(s) to Analyze _(User can edit)_**: Define which variables (age, length, slope, etc.) to include.

- **Merge and Align Data**: Combine information from PIPES, CCTV, and DEFECTS using PIPE_ID.

- **Create color map for defects**: Automatically generate a consistent color palette for defect types.



## 6) Dataset Description

This section provides a comprehensive overview of the datasets used in the analysis. It aims to summarize and visualize the main characteristics of the sewer network, the subset of pipes inspected by CCTV, and their representativeness. Three main functions are used in this stage:

 - `combined_summary_two_tables()`

Compares two datasets (typically the full network and the CCTV-inspected subset) by generating descriptive statistics for both. It provides an overview of how the inspected sample differs from the overall network in terms of key numerical (e.g., pipe length, slope, depth) and categorical factors (e.g., material, sewer type). The outputs help identify potential sampling biases.

 - `plot_ecdf_by_material_overlay()`

Visualizes the cumulative distribution of different variables (such as length, slope, or depth) for the complete network and the CCTV sample, broken down by material. By overlaying both distributions, it highlights whether certain materials or property ranges are overrepresented in the inspections. It is particularly useful for assessing the representativeness of the inspected dataset.

 - `plot_boxplots_grid()`

Gnerates a grid of boxplots that display the distribution of numeric variables grouped by material. It provides a clear visual comparison of how properties such as pipe diameter, length, slope, or depth vary across materials.


## 7) Defect Properties Analysis

In this section, the focus shifts from describing the overall network to analyzing the defects observed during CCTV inspections.
The goal is to understand how defects are distributed, how frequent they are, and where they tend to occur along and around the pipe wall.
The main analyses are grouped into two categories:

### 7.1) Defects per Kilometer and per Pipe
- `plot_defects_per_km()`

Computes and visualizes the number of defects per kilometer for each pipe material.
It normalizes the total number of observed defects by the inspected pipe length, allowing a fair comparison across materials.
The resulting bar chart helps identify which materials have the highest defect density.

- `plot_defect_counts_per_pipe()`

Calculates the average number of defects per pipe and plots them by material.
It complements the per-kilometer analysis, showing whether certain materials consistently accumulate more defects per pipe, regardless of their length.
Together, both plots summarize the overall defect intensity and frequency across materials.

### 7.2) Defect Properties and Correlation

- `plot_defects_stacked_with_others()`

Creates two stacked bar charts showing the composition of defect types by material.
The top chart includes only the most frequent defects (above a user-defined threshold), while less common defects are grouped under “Others.”
The bottom chart expands the “Others” category, providing insight into the rare defect types.
This visualization clarifies which defect types dominate each material.

- `plot_defect_size_bars_()`

Examines the size distribution of defects (Small, Medium, Large) for each material.
It displays a set of horizontal stacked bars—one per material—where the width of each segment represents the relative proportion of each defect size.
The chart helps compare whether some materials tend to experience larger or smaller defects.

- `plot_defect_heatmaps_longitudinal()`

Produces heatmaps showing the longitudinal position of defects along the pipes.
Each defect type forms a row, and the horizontal axis represents normalized pipe length (0–1).
The color intensity indicates the frequency of defects at each position, allowing users to see whether certain defect types occur more often near pipe ends or mid-sections.

- `plot_defect_density_extent_horizontal()`

Analyzes the extent of each defect along the pipe by considering both its start position and normalized length.
It accumulates all defect intervals and visualizes them as a density heatmap, where darker colors indicate zones where defects overlap.
This approach highlights whether certain defect types tend to spread over longer sections or concentrate in specific regions.

- `plot_defect_position_heatmaps()`

Visualizes the circumferential position of defects, using clock references (1–12 o’clock).
It produces heatmaps for each material, showing where around the pipe wall different defect types tend to occur (e.g., roots near the invert, cracks near the crown).
It provides a clear view of how defects are distributed around the pipe circumference.

- `plot_defect_type_correlation()`

Explores how defect types co-occur within the same pipe.
It generates a correlation matrix showing which defects often appear together, suggesting possible causal or structural relationships (e.g., joint faults associated with infiltration or cracking).
This analysis supports the identification of combined failure patterns and dependencies between defect mechanisms.



## 8) How to run the code
To execute the analysis, simply run the script `DEFECT_SEWER_ANALYSIS.py.`
When the code starts, a window will automatically appear asking you to select the Excel file that contains the input data.

Once the file is selected, the program will:

- Validate the input data and check for missing or inconsistent values.

- Generate descriptive summaries of the dataset.

- Analyze the properties and distributions of defects.

- Save all results — including figures and summary tables — inside a folder named `Results`.

No additional user interaction is required beyond selecting the file.
All output files (plots and Excel summaries) are automatically generated.
